"""Evaluation results and the SegEvalKit results format.

A results folder written by `EvaluationResult.save` is the library's
standard output. Every file is plain CSV/JSON so that it can be read without
SegEvalKit:

| File | Contents |
|---|---|
| ``per_case.csv`` | long/tidy: ``case_id, label, metric, value`` (one row per number) |
| ``per_case_wide.csv`` | one row per ``(case_id, label)``, one column per metric, plus ``ref_empty``, ``pred_empty``, ``ref_volume_ml``, ``pred_volume_ml`` |
| ``lesions.csv`` | one row per reference lesion / false-positive component |
| ``summary.csv`` | per ``(label, metric)``: n, mean, std, median, IQR, min, max, 95 % bootstrap CI of the mean, number of NaN |
| ``meta.json`` | provenance: version, full configuration, paths, labels, timing, missing / failed cases |

Descriptive per-case flags are stored in the long table with a leading
underscore (``_ref_empty``...) so they never mix with metrics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = ["EvaluationResult", "load_results"]

RESULTS_FORMAT_VERSION = 1


@dataclass
class EvaluationResult:
    """Container for the outcome of an evaluation.

    Attributes:
        long: Tidy table ``case_id, label, metric, value`` (metrics and ``_`` flags).
        lesions: Per-lesion table (may be empty).
        meta: Provenance dictionary.
    """

    long: pd.DataFrame
    lesions: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------- construction
    @classmethod
    def from_rows(cls, rows: List[dict], lesions: Optional[List[dict]] = None,
                  meta: Optional[dict] = None) -> "EvaluationResult":
        long = pd.DataFrame(rows, columns=["case_id", "label", "metric", "value"])
        if len(long):
            long = long.sort_values(["case_id"], kind="stable").reset_index(drop=True)
        return cls(long, pd.DataFrame(lesions or []), dict(meta or {}))

    # --------------------------------------------------------------- views
    @property
    def name(self) -> str:
        return self.meta.get("name") or Path(str(self.meta.get("pred", "result"))).name

    @property
    def metrics(self) -> List[str]:
        return [m for m in pd.unique(self.long["metric"]) if not str(m).startswith("_")]

    @property
    def labels(self) -> List[str]:
        return list(pd.unique(self.long["label"]))

    @property
    def cases(self) -> List[str]:
        return list(pd.unique(self.long["case_id"]))

    def per_case(self, include_flags: bool = False) -> pd.DataFrame:
        """The long table (optionally with the ``_`` descriptive flags)."""
        if include_flags:
            return self.long.copy()
        return self.long[~self.long["metric"].str.startswith("_")].reset_index(drop=True)

    def wide(self) -> pd.DataFrame:
        """One row per ``(case_id, label)``, one column per metric/flag."""
        w = self.long.pivot_table(index=["case_id", "label"], columns="metric", values="value",
                                  aggfunc="first", dropna=False)
        w.columns.name = None
        w = w.rename(columns=lambda c: c[1:] if str(c).startswith("_") else c)
        order = [m for m in self.metrics if m in w.columns]
        rest = [c for c in w.columns if c not in order]
        return w[order + rest].reset_index()

    def values(self, metric: str, label: Optional[str] = None) -> pd.Series:
        """Per-case values of one metric (indexed by case id)."""
        d = self.long[self.long["metric"] == metric]
        if label is not None:
            d = d[d["label"] == label]
        return d.set_index("case_id")["value"] if label is not None else d.set_index(["case_id", "label"])["value"]

    def summary(self, ci: float = 0.95, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
        """Per ``(label, metric)`` descriptive statistics with a bootstrap CI of the mean."""
        from .stats import summarize

        return summarize(self.per_case(), ci=ci, n_boot=n_boot, seed=seed)

    def filter(self, labels: Optional[Sequence[str]] = None, metrics: Optional[Sequence[str]] = None,
               cases: Optional[Sequence[str]] = None) -> "EvaluationResult":
        """Subset by labels / metrics / cases (flags are kept for the selected rows)."""
        d = self.long
        if labels is not None:
            d = d[d["label"].isin(labels)]
        if cases is not None:
            d = d[d["case_id"].isin(cases)]
        if metrics is not None:
            d = d[d["metric"].isin(list(metrics)) | d["metric"].str.startswith("_")]
        les = self.lesions
        if len(les):
            if labels is not None:
                les = les[les["label"].isin(labels)]
            if cases is not None:
                les = les[les["case_id"].isin(cases)]
        return EvaluationResult(d.reset_index(drop=True), les.reset_index(drop=True), dict(self.meta))

    def worst_cases(self, metric: str, label: str, k: int = 5) -> pd.DataFrame:
        """The ``k`` worst cases for ``metric`` on ``label`` (direction-aware)."""
        from .metrics import get_metric

        info = get_metric(metric)
        v = self.values(metric, label).dropna()
        if info.better == "higher":
            v = v.sort_values()
        elif info.better == "lower":
            v = v.sort_values(ascending=False)
        else:
            v = v.reindex(v.abs().sort_values(ascending=False).index)
        return v.head(k).rename(metric).reset_index()

    # -------------------------------------------------------------- persist
    def save(self, out_dir: Union[str, os.PathLike], summary: bool = True) -> Path:
        """Write the standard results folder (see module docstring)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.long.to_csv(out / "per_case.csv", index=False)
        self.wide().to_csv(out / "per_case_wide.csv", index=False)
        if len(self.lesions):
            self.lesions.to_csv(out / "lesions.csv", index=False)
        if summary and len(self.long):
            self.summary().to_csv(out / "summary.csv", index=False)
        meta = {"results_format": RESULTS_FORMAT_VERSION, **self.meta}
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        return out

    def __repr__(self) -> str:
        return (f"EvaluationResult(name={self.name!r}, cases={len(self.cases)}, labels={self.labels[:6]}"
                f"{'...' if len(self.labels) > 6 else ''}, metrics={len(self.metrics)})")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def load_results(path: Union[str, os.PathLike]) -> EvaluationResult:
    """Load a results folder written by `EvaluationResult.save`."""
    p = Path(path)
    long = pd.read_csv(p / "per_case.csv", dtype={"case_id": str, "label": str})
    les = pd.read_csv(p / "lesions.csv", dtype={"case_id": str}) if (p / "lesions.csv").exists() else pd.DataFrame()
    meta = json.loads((p / "meta.json").read_text()) if (p / "meta.json").exists() else {}
    return EvaluationResult(long, les, meta)
