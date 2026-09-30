"""Self-contained HTML evaluation report.

The report is a single HTML file (figures embedded as PNG) containing: run
provenance and configuration, warnings (missing / failed cases), per-structure
summary tables with direction arrows and 95 % CIs, distribution / size /
correlation / volume / failure-mode figures, a worst-case gallery with error
overlays, and a glossary with the definition and reference of every metric
used, so a reader never has to guess what a number means.

Examples:
    >>> from segevalkit.report import build_report
    >>> build_report(result, "report.html", image_source="imagesTr/")
"""

from __future__ import annotations

import base64
import html
import io
import math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..metrics import get_metric

__all__ = ["build_report", "fig_to_base64"]

_TEMPLATE = Path(__file__).with_name("template.html")


def fig_to_base64(fig, dpi: int = 150) -> str:
    """Encode a matplotlib figure as a base64 PNG data string and close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _short_path(p) -> str:
    """Last two path components (the report is meant to be shared; hide machine-specific prefixes)."""
    parts = Path(str(p)).parts
    return str(Path(*parts[-2:])) if len(parts) > 2 else str(p)


def _fmt(v: float, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    a = abs(v)
    s = f"{v:.3f}" if a < 10 else (f"{v:.2f}" if a < 100 else f"{v:.1f}")
    return s


def _summary_tables(result, metrics: Sequence[str]) -> List[Dict]:
    summ = result.summary()
    tables = []
    for label in result.labels:
        rows = []
        s = summ[summ["label"] == label].set_index("metric")
        for m in metrics:
            if m not in s.index:
                continue
            r = s.loc[m]
            info = get_metric(m)
            rows.append({
                "metric": info.display, "abbr": info.abbr, "arrow": info.arrow, "unit": info.unit,
                "mean": _fmt(r["mean"]), "std": _fmt(r["std"]), "median": _fmt(r["median"]),
                "iqr": f"{_fmt(r['q1'])} – {_fmt(r['q3'])}",
                "ci": f"{_fmt(r['ci_low'])} – {_fmt(r['ci_high'])}", "n": int(r["n"]), "n_nan": int(r["n_nan"]),
            })
        tables.append({"label": label, "rows": rows})
    return tables


def _glossary(metrics: Sequence[str]) -> List[Dict]:
    out = []
    for m in metrics:
        info = get_metric(m)
        eq = (info.fn.__doc__ or "").strip().split("\n\n")[0] if info.fn else ""
        rng = info.value_range
        out.append({"name": info.display, "key": m, "abbr": info.abbr, "family": info.family,
                    "summary": info.summary, "reference": info.reference,
                    "equation": eq if eq.startswith("$$") else "",
                    "range": f"[{_fmt(rng[0])}, {'∞' if not math.isfinite(rng[1]) else _fmt(rng[1])}]",
                    "better": {"higher": "higher is better", "lower": "lower is better",
                               "zero": "0 is ideal", "none": "descriptive"}[info.better],
                    "unit": info.unit or "–"})
    return out


def _figures(result, metrics: Sequence[str], max_labels: int) -> List[Dict]:
    from .. import plotting as P

    figs = []
    labels = result.labels[:max_labels]
    for m in [x for x in ("dice", "nsd", "hd95", "assd", "cldice", "lesion_f1") if x in metrics][:4]:
        figs.append({"title": f"Distribution of {get_metric(m).abbr}",
                     "caption": "Half-violin: distribution; box: median and IQR; dots: individual cases.",
                     "png": fig_to_base64(P.metric_distribution(result, m, labels=labels))})
    main = next((m for m in ("dice", "nsd", "iou") if m in metrics), None)
    for label in labels[:3]:
        if main:
            figs.append({"title": f"{get_metric(main).abbr} vs structure size: {label}",
                         "caption": "Binned median trend; small structures typically score lower for the same boundary error.",
                         "png": fig_to_base64(P.metric_vs_size(result, main, label=label))})
        if "hd95" in metrics and "dice" in metrics:
            figs.append({"title": f"Failure modes: {label}",
                         "caption": "Good overlap with large HD95 suggests distant false positives; poor overlap with "
                                    "small HD95 suggests systematic under-segmentation.",
                         "png": fig_to_base64(P.failure_quadrants(result, label))})
        figs.append({"title": f"Volume agreement: {label}", "caption": "Dashed line: identity.",
                     "png": fig_to_base64(P.volume_agreement(result, label))})
    corr_metrics = [m for m in metrics if get_metric(m).family != "volume"][:14]
    if len(corr_metrics) >= 3 and len(result.cases) >= 5:
        figs.append({"title": "Metric correlation",
                     "caption": "Spearman correlation across cases and structures: highly correlated metrics carry "
                                "redundant information; weakly correlated ones measure different failure modes.",
                     "png": fig_to_base64(P.metric_correlation(result, corr_metrics))})
    if len(result.lesions) and (result.lesions["kind"] == "ref").any():
        figs.append({"title": "Lesion detection by size", "caption": "Fraction of reference components detected.",
                     "png": fig_to_base64(P.detection_by_size(result))})
    return figs


def _gallery(result, image_source, metric: str, k: int, window) -> List[Dict]:
    """Worst-case overlays; needs the original prediction/reference folders from the metadata."""
    from ..io import LabelSpec, Source, load_volume
    from ..viz import error_overlay, triplanar

    meta = result.meta
    if "pred" not in meta or "ref" not in meta:
        return []
    try:
        pred_src, ref_src = Source(meta["pred"]), Source(meta["ref"])
        img_src = Source(image_source, kind="image") if image_source else None
    except Exception:
        return []
    labels = {d["name"]: LabelSpec(d["name"], tuple(d["ref_values"]), tuple(d["pred_values"]),
                                   d.get("ref_file"), d.get("pred_file")) for d in meta.get("labels", [])}
    items = []
    for label in result.labels[:3]:
        if label not in labels:
            continue
        for _, row in result.worst_cases(metric, label, k=k).iterrows():
            cid = row["case_id"]
            try:
                ref, rv = ref_src.load_mask(cid, labels[label], "ref")
                pred = np.zeros_like(ref)
                if cid in pred_src.case_ids:
                    p, _ = pred_src.load_mask(cid, labels[label], "pred")
                    pred = p if p is not None and p.shape == ref.shape else pred
                img = None
                if img_src is not None and cid in img_src.case_ids:
                    iv = img_src.load(cid) if img_src.layout != "per_structure" else None
                    img = iv.data if iv is not None and iv.shape == ref.shape else None
                fig = triplanar(img, pred, ref, affine=rv.affine, window=window,
                                title=f"{cid} · {label} · {get_metric(metric).abbr} = {row[metric]:.3f}")
                items.append({"title": f"{cid} · {label}", "png": fig_to_base64(fig, dpi=110)})
            except Exception as exc:  # pragma: no cover - best effort
                items.append({"title": f"{cid} · {label}", "png": None, "error": str(exc)})
    return items


def build_report(result: "EvaluationResult", path: Union[str, Path], *, title: Optional[str] = None,
                 metrics: Optional[Sequence[str]] = None, image_source: Union[str, Path, None] = None,
                 gallery_metric: str = "dice", gallery_k: int = 3, window="abdomen",
                 max_labels: int = 8, gallery: bool = True) -> Path:
    """Write a self-contained HTML report for an `EvaluationResult`.

    Args:
        result: The evaluation result.
        path: Output ``.html`` path.
        metrics: Metrics to show (default: all metrics in the result).
        image_source: Folder of images (same layout rules as labels) for the
            worst-case gallery; without it overlays are drawn on black.
        gallery_metric: Metric used to pick the worst cases.
    """
    from jinja2 import Template

    metrics = [m for m in (metrics or result.metrics) if m in result.metrics]
    meta = result.meta
    cfg = meta.get("config", {})
    warnings_ = []
    if meta.get("missing_pred"):
        mp = meta["missing_pred"]
        warnings_.append(f"{len(mp)} reference case(s) had no prediction and were scored as empty: "
                         + ", ".join(map(str, mp[:10])) + (" …" if len(mp) > 10 else ""))
    if meta.get("errors"):
        warnings_.append(f"{len(meta['errors'])} case(s) failed: " + ", ".join(list(meta["errors"])[:10]))
    gal = []
    if gallery and gallery_metric in metrics:
        gal = _gallery(result, image_source, gallery_metric, gallery_k, window)
    ctx = {
        "title": title or f"Segmentation evaluation: {result.name}",
        "meta": {**meta, "pred": _short_path(meta["pred"]) if meta.get("pred") else None,
                 "ref": _short_path(meta["ref"]) if meta.get("ref") else None},
        "cfg": cfg,
        "n_cases": len(result.cases), "labels": result.labels, "n_metrics": len(metrics),
        "params": {k: v for k, v in cfg.get("params", {}).items()},
        "warnings": warnings_,
        "tables": _summary_tables(result, metrics),
        "figures": _figures(result, metrics, max_labels),
        "gallery": gal, "gallery_metric": get_metric(gallery_metric).display if gallery_metric in metrics else "",
        "glossary": _glossary(metrics),
    }
    html_text = Template(_TEMPLATE.read_text(encoding="utf-8"), autoescape=True).render(**ctx, esc=html.escape)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path
