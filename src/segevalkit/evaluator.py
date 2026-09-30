"""The dataset-level evaluation engine.

`Evaluator` turns *(prediction folder, reference folder, label spec)*
into a tidy, reproducible `EvaluationResult`:

```python
from segevalkit import Evaluator

ev = Evaluator(labels={"liver": 1, "tumour": 2},
               metrics=["default", "lesion_f1"],
               params={"nsd": {"tolerance_mm": 2.0}})
res = ev.evaluate("preds/", "labelsTr/", n_workers=8)
res.summary()          # per-label mean / median / CI table
res.save("eval_out/")  # standard SegEvalKit results folder
```

Every choice that changes a number (empty-mask policy, connectivity,
tolerances, alignment handling) is an explicit argument and is written to the
results' provenance, so any table produced with SegEvalKit can be regenerated.
"""

from __future__ import annotations

import logging
import math
import os
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from . import __version__
from .io.cases import Case, Source, discover_cases
from .io.labels import LabelSpec, labels_from_map, parse_labels
from .io.volume import Volume, check_alignment, resample_to
from .metrics import EmptyPolicy, PairContext, get_metric, lesion_table, match_instances, resolve_metrics
from .results import EvaluationResult

__all__ = ["Evaluator", "EvalConfig"]

log = logging.getLogger("segevalkit")

_DETECTION = {"lesion_recall", "lesion_precision", "lesion_f1", "panoptic_quality", "split_count", "merge_count",
              "lesionwise_dice", "lesionwise_hd95", "lesionwise_nsd", "lesion_ap",
              "lesion_count_difference", "false_positive_lesions",
              "false_negative_lesions"}


@dataclass
class EvalConfig:
    """Serializable evaluation configuration (everything that changes a number)."""

    metrics: List[str] = field(default_factory=lambda: resolve_metrics("default"))
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    both_empty: str = "best"
    one_empty_distance: Union[str, float] = "worst"
    connectivity: int = 26
    min_lesion_voxels: int = 0
    alignment: str = "strict"
    missing_pred: str = "empty"
    lesion_table: bool = True
    lesion_score: str = "max"
    device: str = "cpu"

    @property
    def empty(self) -> EmptyPolicy:
        return EmptyPolicy(self.both_empty, self.one_empty_distance)


class Evaluator:
    """Evaluate segmentations over labels and cases.

    Args:
        labels: Structures to evaluate (see `segevalkit.io.parse_labels`);
            ``None`` evaluates every non-zero value found in the reference
            (multi-label layouts) or every reference structure file
            (per-structure layouts).
        metrics: Metric names / aliases / set names (``"default"``, ``"all"``...).
        params: Global per-metric keyword overrides, e.g.
            ``{"nsd": {"tolerance_mm": 1.0}}``. Per-label overrides in the
            label spec take precedence.
        empty: `EmptyPolicy` for empty masks.
        device: ``"cpu"`` or ``"cuda"``/``"cuda:N"`` (surface distances and
            confusion counts on the GPU).
        connectivity: Connectivity for lesion/instance components (6/18/26).
        min_lesion_voxels: Ignore connected components smaller than this for
            detection metrics.
        alignment: What to do when prediction and reference grids differ:
            ``"strict"`` (error), ``"resample"`` (nearest-neighbour resample the
            prediction onto the reference grid) or ``"ignore"`` (only require
            equal shapes; use when headers are known to be unreliable).
        missing_pred: ``"empty"`` scores a missing prediction as an empty mask
            (the challenge convention; a model cannot skip hard cases) or
            ``"skip"`` drops the case.
        lesion_table: Also collect per-lesion rows when detection metrics run.
        lesion_score: Confidence of each predicted lesion in the lesion table:
            ``"max"`` (default) or ``"mean"`` probability inside the component
            when a probability source is given, else its volume in mL. Used by
            `segevalkit.stats.froc` and `segevalkit.stats.lesion_pr`.
    """

    def __init__(
        self,
        labels: Union[Mapping, Sequence[str], None] = None,
        metrics: Union[str, Iterable[str]] = "default",
        params: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        empty: EmptyPolicy = EmptyPolicy(),
        device: str = "cpu",
        connectivity: int = 26,
        min_lesion_voxels: int = 0,
        alignment: str = "strict",
        missing_pred: str = "empty",
        lesion_table: bool = True,
        lesion_score: str = "max",
    ) -> None:
        if alignment not in ("strict", "resample", "ignore"):
            raise ValueError("alignment must be 'strict', 'resample' or 'ignore'")
        if missing_pred not in ("empty", "skip"):
            raise ValueError("missing_pred must be 'empty' or 'skip'")
        if lesion_score not in ("max", "mean"):
            raise ValueError("lesion_score must be 'max' or 'mean'")
        self.labels: List[LabelSpec] = parse_labels(labels)
        names = resolve_metrics(metrics)
        for n in (params or {}):
            get_metric(n)  # fail fast on typos
        self.config = EvalConfig(
            metrics=names, params={k: dict(v) for k, v in (params or {}).items()},
            both_empty=empty.both_empty, one_empty_distance=empty.one_empty_distance,
            connectivity=connectivity, min_lesion_voxels=min_lesion_voxels,
            alignment=alignment, missing_pred=missing_pred, lesion_table=lesion_table,
            lesion_score=lesion_score, device=device,
        )

    # --------------------------------------------------------------- arrays
    def evaluate_arrays(
        self,
        pred: np.ndarray,
        ref: np.ndarray,
        spacing: Optional[Sequence[float]] = None,
        *,
        prob: Optional[Mapping[str, np.ndarray]] = None,
        case_id: str = "case",
    ) -> EvaluationResult:
        """Evaluate one in-memory multi-label (or binary) pair.

        ``prob`` maps label names to foreground probability maps.
        """
        pred = np.asarray(pred)
        ref = np.asarray(ref)
        labels = self.labels or labels_from_map(ref)
        rows, lesions = [], []
        for lab in labels:
            p = _mask_from_array(pred, lab, "pred")
            g = _mask_from_array(ref, lab, "ref")
            pr = None if prob is None else prob.get(lab.name)
            r, les = _score_label(case_id, lab, p, g, spacing, pr, self.config)
            rows.extend(r)
            lesions.extend(les)
        return EvaluationResult.from_rows(rows, lesions, meta=self._meta(n_cases=1))

    # --------------------------------------------------------------- folders
    def evaluate(
        self,
        pred: Union[str, os.PathLike, Source],
        ref: Union[str, os.PathLike, Source],
        *,
        prob: Union[str, os.PathLike, Source, None] = None,
        cases: Optional[Sequence[str]] = None,
        n_workers: int = 1,
        progress: bool = True,
        out_dir: Union[str, os.PathLike, None] = None,
        name: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate every reference case against its prediction.

        Args:
            pred: Prediction folder (or `Source`).
            ref: Reference folder (or ``Source``). Defines the case list.
            prob: Optional folder of per-structure probability maps
                (``<case>/<label>.nii.gz``, float in [0, 1]).
            cases: Restrict to these case ids.
            n_workers: Parallel worker processes (each loads and scores
                whole cases). With a CUDA device, 2-4 workers keep the GPU busy.
            progress: Show a progress bar.
            out_dir: If given, the result is saved there as well.
            name: Optional method/run name stored in the result metadata.
        """
        pred_src = pred if isinstance(pred, Source) else Source(pred)
        ref_src = ref if isinstance(ref, Source) else Source(ref)
        prob_src = None
        if prob is not None:
            prob_src = prob if isinstance(prob, Source) else Source(prob, layout="per_structure", subdir="", kind="prob")
        case_list, report = discover_cases(pred_src, ref_src, cases=cases)
        if report["missing_pred"]:
            msg = f"{len(report['missing_pred'])} reference cases have no prediction"
            if self.config.missing_pred == "skip":
                case_list = [c for c in case_list if c.has_pred]
                msg += " (skipped)"
            else:
                msg += " (scored as empty predictions)"
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        labels = self.labels or _infer_labels(ref_src, case_list[0].case_id)
        t0 = time.time()
        rows: List[dict] = []
        lesions: List[dict] = []
        errors: Dict[str, str] = {}
        jobs = [(c, pred_src, ref_src, prob_src, labels, self.config) for c in case_list]
        bar = _progress(len(jobs), progress, desc=name or "evaluating")
        if n_workers <= 1:
            for job in jobs:
                r, les, err = _run_case(*job)
                rows += r
                lesions += les
                if err:
                    errors[job[0].case_id] = err
                bar.update(1)
        else:
            ctx = get_context("spawn" if self.config.device != "cpu" else "fork")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futs = {pool.submit(_run_case, *job): job[0].case_id for job in jobs}
                for fut in as_completed(futs):
                    r, les, err = fut.result()
                    rows += r
                    lesions += les
                    if err:
                        errors[futs[fut]] = err
                    bar.update(1)
        bar.close()
        if errors:
            warnings.warn(f"{len(errors)} cases failed; see result.meta['errors']", RuntimeWarning, stacklevel=2)
        meta = self._meta(
            n_cases=len(case_list), pred=str(pred_src.root), ref=str(ref_src.root),
            pred_layout=pred_src.layout, ref_layout=ref_src.layout,
            labels=[asdict(lab) for lab in labels], missing_pred=report["missing_pred"],
            extra_pred=report["extra_pred"], errors=errors, seconds=round(time.time() - t0, 2),
            name=name,
        )
        res = EvaluationResult.from_rows(rows, lesions, meta=meta)
        if out_dir is not None:
            res.save(out_dir)
        return res

    def _meta(self, **extra) -> Dict[str, Any]:
        import platform

        return {"segevalkit_version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "python": platform.python_version(), "config": asdict(self.config), **extra}


# ---------------------------------------------------------------- internals
def _mask_from_array(a: np.ndarray, lab: LabelSpec, side: str) -> np.ndarray:
    values = lab.values(side)
    if not values:
        return a != 0
    return a == values[0] if len(values) == 1 else np.isin(a, values)


def _infer_labels(ref: Source, case_id: str) -> List[LabelSpec]:
    if ref.layout == "per_structure":
        return [LabelSpec(s) for s in ref.structures(case_id)]
    vol = ref.load(case_id)
    return labels_from_map(vol.data)


def _score_label(case_id, lab: LabelSpec, pred, ref, spacing, prob, cfg: EvalConfig):
    params = {k: dict(v) for k, v in cfg.params.items()}
    for k, v in lab.params.items():
        params.setdefault(k, {}).update(v)
    ctx = PairContext(pred, ref, spacing, prob, device=cfg.device, empty=cfg.empty,
                      connectivity=cfg.connectivity, min_component_voxels=cfg.min_lesion_voxels)
    rows = []
    metric_names = resolve_metrics(lab.metrics) if lab.metrics else cfg.metrics
    for name in metric_names:
        info = get_metric(name)
        if "probabilities" in info.requires and prob is None:
            continue
        try:
            val = float(info(ctx, **params.get(name, {})))
        except Exception as exc:
            log.warning("case %s label %s metric %s failed: %s", case_id, lab.name, name, exc)
            val = math.nan
        rows.append({"case_id": case_id, "label": lab.name, "metric": name, "value": val})
    vox_ml = ctx.voxel_volume_mm3 / 1000.0
    # Descriptive flags and raw counts (prefixed "_"): the counts make dataset-level
    # pooled ("micro") metrics possible, e.g. aggregated Dice = 2ΣTP / (2ΣTP + ΣFP + ΣFN).
    flags = [("ref_empty", float(ctx.ref_empty)), ("pred_empty", float(ctx.pred_empty)),
             ("ref_volume_ml", (ctx.tp + ctx.fn) * vox_ml), ("pred_volume_ml", (ctx.tp + ctx.fp) * vox_ml),
             ("tp", float(ctx.tp)), ("fp", float(ctx.fp)), ("fn", float(ctx.fn)), ("tn", float(ctx.tn))]
    lesions = []
    if cfg.lesion_table and _DETECTION.intersection(metric_names):
        m = match_instances(ctx, criterion="overlap")
        flags += [("n_ref_lesions", float(m.n_ref)), ("n_pred_lesions", float(m.n_pred)),
                  ("tp_ref_lesions", float(m.tp_ref)), ("tp_pred_lesions", float(m.tp_pred))]
        for r in lesion_table(ctx, score=cfg.lesion_score):
            lesions.append({"case_id": case_id, "label": lab.name, **r})
    for key, val in flags:
        rows.append({"case_id": case_id, "label": lab.name, "metric": f"_{key}", "value": val})
    return rows, lesions


def _run_case(case: Case, pred_src: Source, ref_src: Source, prob_src: Optional[Source],
              labels: List[LabelSpec], cfg: EvalConfig):
    rows, lesions = [], []
    try:
        for lab in labels:
            ref, ref_vol = ref_src.load_mask(case.case_id, lab, "ref")
            if ref is None:
                log.warning("case %s: reference structure %s missing; skipped", case.case_id, lab.name)
                continue
            pred = None
            pred_vol = None
            if case.has_pred:
                pred, pred_vol = pred_src.load_mask(case.case_id, lab, "pred")
            if pred is None:
                pred = np.zeros_like(ref)
            elif cfg.alignment != "ignore" or pred.shape != ref.shape:
                issues = check_alignment(pred_vol, ref_vol) if pred_vol is not None else []
                if issues and cfg.alignment == "resample":
                    pred = resample_to(Volume(pred.astype(np.uint8), pred_vol.spacing, pred_vol.affine),
                                       ref_vol).data.astype(bool)
                elif issues and (cfg.alignment == "strict" or pred.shape != ref.shape):
                    raise ValueError(f"prediction/reference grids differ for {lab.name}: {'; '.join(issues)} "
                                     "(use alignment='resample' or 'ignore')")
            prob = None
            if prob_src is not None and case.case_id in prob_src._index:
                prob, _ = prob_src.load_mask(case.case_id, lab, "pred")
            r, les = _score_label(case.case_id, lab, pred, ref, ref_vol.spacing, prob, cfg)
            rows += r
            lesions += les
        return rows, lesions, None
    except Exception:
        return rows, lesions, traceback.format_exc(limit=3)


class _NullBar:
    def update(self, n=1):
        pass

    def close(self):
        pass


class _RichBar:
    """Purple rich progress bar with the same update/close interface as tqdm."""

    def __init__(self, total: int, desc: str):
        from ._console import progress

        self._p = progress()
        self._p.start()
        self._task = self._p.add_task(desc, total=total)

    def update(self, n=1):
        self._p.advance(self._task, n)

    def close(self):
        self._p.stop()


def _progress(total: int, enabled: bool, desc: str):
    if not enabled:
        return _NullBar()
    try:
        return _RichBar(total, desc)
    except ImportError:  # pragma: no cover
        from tqdm.auto import tqdm

        return tqdm(total=total, desc=desc, unit="case")
