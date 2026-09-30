"""The ``segevalkit`` command line.

.. code-block:: console

    segevalkit evaluate --pred preds/ --ref labelsTr/ --labels liver=1,tumour=2 --out eval/ --report
    segevalkit evaluate --config eval.yaml
    segevalkit evaluate --dataset pants --pred preds/ --ref PanTS/LabelTe --out eval/
    segevalkit report eval/ --images imagesTr/
    segevalkit compare eval_A/ eval_B/ --out compare/
    segevalkit rank eval_A/ eval_B/ eval_C/ --metric dice --label liver
    segevalkit metrics
    segevalkit recommend --structure small_lesion --multi-instance
    segevalkit datasets [NAME]
    segevalkit visualize --pred p.nii.gz --ref g.nii.gz --image ct.nii.gz --label 1 --out fig.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_label_arg(s: Optional[str]):
    """``liver=1,tumour=2,kidney=3+4`` or ``liver,pancreas`` or a path to JSON/YAML."""
    if s is None:
        return None
    p = Path(s)
    if p.exists():
        return _load_mapping(p).get("labels", _load_mapping(p))
    if "=" not in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    out: Dict[str, Any] = {}
    for item in s.split(","):
        name, _, v = item.partition("=")
        out[name.strip()] = [int(x) for x in v.split("+")]
    return out


def _load_mapping(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text()
    if str(path).endswith((".yaml", ".yml")):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def _empty_policy(v):
    from .metrics import EmptyPolicy

    if v is None:
        return EmptyPolicy()
    if isinstance(v, dict):
        return EmptyPolicy(**v)
    return EmptyPolicy.preset(str(v))


def cmd_evaluate(a) -> int:
    from . import Evaluator
    from .io import Source

    cfg: Dict[str, Any] = _load_mapping(Path(a.config)) if a.config else {}
    preset = None
    if a.dataset or cfg.get("dataset"):
        from .datasets import get_dataset

        preset = get_dataset(a.dataset or cfg["dataset"])
    labels = _parse_label_arg(a.labels) if a.labels else cfg.get("labels", preset.labels if preset else None)
    metrics = a.metrics or cfg.get("metrics") or (preset.metrics if preset else "default")
    params = dict(preset.params) if preset else {}
    params.update(cfg.get("params", {}))
    if a.nsd_tolerance is not None:
        params.setdefault("nsd", {})["tolerance_mm"] = a.nsd_tolerance
    pred = a.pred or cfg.get("pred")
    ref = a.ref or cfg.get("ref")
    if not pred or not ref:
        print("error: --pred and --ref (or a config with pred/ref) are required", file=sys.stderr)
        return 2
    out = a.out or cfg.get("out") or "segevalkit_results"
    from ._console import banner, console, rule

    banner("evaluate")
    ev = Evaluator(
        labels=labels, metrics=metrics, params=params,
        empty=_empty_policy(a.empty_policy or cfg.get("empty_policy") or (preset.empty_policy if preset else None)),
        device=a.device or cfg.get("device", "cpu"),
        connectivity=int(cfg.get("connectivity", a.connectivity)),
        min_lesion_voxels=int(cfg.get("min_lesion_voxels", a.min_lesion_voxels)),
        alignment=a.alignment or cfg.get("alignment", "strict"),
        missing_pred=cfg.get("missing_pred", "empty"),
    )
    ref_kw = {}
    if preset and preset.ref_layout:
        ref_kw = dict(layout=preset.ref_layout, file=preset.ref_file, subdir=preset.ref_subdir)
    ref_src = Source(ref, **{k: v for k, v in ref_kw.items() if v is not None})
    pred_src = Source(pred, layout=cfg.get("pred_layout", "auto"), file=cfg.get("pred_file"),
                      subdir=cfg.get("pred_subdir"))
    console.print(f"[sek.muted]predictions[/]  [sek.key]{pred_src.root}[/] [sek.muted]({pred_src.layout}, "
                  f"{len(pred_src.case_ids)} cases)[/]")
    console.print(f"[sek.muted]reference  [/]  [sek.key]{ref_src.root}[/] [sek.muted]({ref_src.layout}, "
                  f"{len(ref_src.case_ids)} cases)[/]")
    console.print(f"[sek.muted]metrics    [/]  {', '.join(ev.config.metrics)}")
    console.print(f"[sek.muted]device     [/]  {ev.config.device}  [sek.muted]workers[/] "
                  f"{int(a.workers or cfg.get('workers', 1))}\n")
    cases = None
    if a.cases:
        cases = [x.strip() for x in Path(a.cases).read_text().split() if x.strip()]
    res = ev.evaluate(pred_src, ref_src, prob=a.prob or cfg.get("prob"), cases=cases,
                      n_workers=int(a.workers or cfg.get("workers", 1)), out_dir=out,
                      name=a.name or cfg.get("name"))
    from ._console import summary_table

    rule("results")
    console.print(summary_table(res.summary(), res.metrics[:7], title=f"{res.name}: {len(res.cases)} cases"))
    if res.meta.get("missing_pred"):
        console.print(f"[sek.warn]⚠ {len(res.meta['missing_pred'])} cases without prediction scored as empty[/]")
    if res.meta.get("errors"):
        console.print(f"[sek.warn]⚠ {len(res.meta['errors'])} cases failed (see meta.json)[/]")
    if a.report or cfg.get("report"):
        from .report import build_report

        with console.status("[sek.brand]building HTML report…", spinner="dots12"):
            rp = build_report(res, Path(out) / "report.html", image_source=a.images or cfg.get("images"))
        console.print(f"[sek.ok]✓[/] report   [sek.key]{rp}[/]")
    console.print(f"[sek.ok]✓[/] results  [sek.key]{Path(out).resolve()}[/]  "
                  f"[sek.muted]({res.meta.get('seconds', 0):.1f} s)[/]")
    return 0


class _pd_opts:
    def __enter__(self):
        import pandas as pd

        self._ctx = pd.option_context("display.width", 160, "display.max_rows", 200)
        self._ctx.__enter__()

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)


def cmd_report(a) -> int:
    from .report import build_report
    from .results import load_results

    from ._console import banner, console

    banner("report", compact=True)
    res = load_results(a.results)
    with console.status("[sek.brand]rendering figures and overlays…", spinner="dots12"):
        out = build_report(res, a.out or Path(a.results) / "report.html", image_source=a.images,
                           gallery_metric=a.gallery_metric, window=a.window)
    console.print(f"[sek.ok]✓[/] report  [sek.key]{out}[/]")
    return 0


def cmd_compare(a) -> int:
    from .results import load_results
    from .stats import compare

    from rich.table import Table

    from ._console import P, banner, console, fmt
    from .metrics import get_metric

    banner("compare", compact=True)
    ra, rb = load_results(a.a), load_results(a.b)
    na, nb = a.name_a or ra.name, a.name_b or rb.name
    df = compare(ra, rb, metrics=a.metrics.split(",") if a.metrics else None, test=a.test, correction=a.correction)
    t = Table(title=f"[sek.brand]{na}[/] vs [sek.brand]{nb}[/]  [sek.muted](paired {a.test}, {a.correction})[/]",
              border_style=P["800"], header_style=f"bold {P['200']}")
    for c in ("structure", "metric", "n", na, nb, "Δ (95% CI)", f"{na} better", "p adj", ""):
        t.add_column(c, justify="left" if c in ("structure", "metric") else "right")
    for _, r in df.iterrows():
        info = get_metric(r["metric"])
        win = r["frac_a_better"]
        sig = "[sek.ok]●[/]" if r.get("significant") else "[sek.muted]○[/]"
        t.add_row(str(r["label"]), f"{info.abbr} [sek.arrow]{info.arrow}[/]", str(int(r["n"])), fmt(r["mean_a"]),
                  fmt(r["mean_b"]), f"{fmt(r['mean_diff'])} [sek.muted]({fmt(r['diff_ci_low'])}, "
                  f"{fmt(r['diff_ci_high'])})[/]", f"{100 * win:.0f}%", f"{r['p_adjusted']:.2g}", sig)
    console.print(t)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(a.out) / "comparison.csv", index=False)
        from . import plotting

        fig = plotting.comparison_forest(df, name_a=a.name_a or ra.name, name_b=a.name_b or rb.name)
        fig.savefig(Path(a.out) / "comparison_forest.png")
        console.print(f"[sek.ok]✓[/] written to [sek.key]{Path(a.out).resolve()}[/]")
    return 0


def cmd_rank(a) -> int:
    from .results import load_results
    from .stats import rank_methods, ranking_stability

    import numpy as np
    from rich.table import Table

    from ._console import P, banner, console, fmt

    banner("rank", compact=True)
    res = {load_results(p).name: load_results(p) for p in a.results}
    t = Table(title=f"[sek.brand]Ranking by {a.metric}[/]" + (f" [sek.muted]· {a.label}[/]" if a.label else ""),
              border_style=P["800"], header_style=f"bold {P['200']}")
    t.add_column("method", style="sek.key")
    t.add_column("aggregate→rank", justify="right")
    t.add_column("mean", justify="right")
    t.add_column("rank→aggregate", justify="right")
    t.add_column("mean rank", justify="right")
    r1 = rank_methods(res, a.metric, a.label, "aggregate-then-rank").set_index("method")
    r2 = rank_methods(res, a.metric, a.label, "rank-then-aggregate").set_index("method")
    medal = {1: "🥇", 2: "🥈", 3: "🥉"}
    for m in r1.sort_values("rank").index:
        k1, k2 = int(r1.loc[m, "rank"]), int(r2.loc[m, "rank"])
        t.add_row(m, f"{medal.get(k1, '')} {k1}", fmt(r1.loc[m, "mean"]), f"{medal.get(k2, '')} {k2}",
                  fmt(r2.loc[m, "mean_rank"]))
    console.print(t)
    with console.status("[sek.brand]bootstrapping rankings…", spinner="dots12"):
        stab = ranking_stability(res, a.metric, a.label, n_boot=a.n_boot)
    tau = float(np.nanmedian(stab["kendall_tau"]))
    console.print(f"bootstrap stability ({a.n_boot} samples): median Kendall τ = [sek.num]{tau:.3f}[/]")
    if a.out:
        from . import plotting

        plotting.ranking_stability_plot(stab).savefig(a.out)
        console.print(f"[sek.ok]✓[/] [sek.key]{a.out}[/]")
    return 0


def cmd_metrics(a) -> int:
    from ._console import banner, console, metric_table, rule
    from .metrics import FAMILIES, list_metrics

    banner(f"{len(list_metrics())} metrics in {len(FAMILIES)} families", compact=True)
    for fam, title in FAMILIES.items():
        if a.family and fam != a.family:
            continue
        ms = list_metrics(fam)
        if not ms:
            continue
        rule(title)
        console.print(metric_table(ms, verbose=a.verbose))
    return 0


def cmd_recommend(a) -> int:
    from .guide import Fingerprint, recommend

    fp = Fingerprint(structure=a.structure, multi_instance=a.multi_instance, boundary_critical=a.boundary_critical,
                     volumetry=a.volumetry, empty_references=a.empty_references, probabilistic=a.probabilistic,
                     fp_fn_asymmetric=a.asymmetric, noisy_reference=a.noisy_reference,
                     tolerance_mm=a.tolerance, ranking=a.ranking)
    r = recommend(fp)
    if a.markdown:
        print(r.to_markdown())
        return 0
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    from ._console import P, banner, console, panel
    from .metrics import get_metric

    banner("metric recommendation", compact=True)
    flags = [k for k, v in vars(fp).items() if v is True]
    console.print(f"[sek.muted]fingerprint[/]  [sek.key]{fp.structure}[/]" + (f"  +  {', '.join(flags)}" if flags else ""))
    t = Table(border_style=P["800"], header_style=f"bold {P['200']}", expand=True)
    t.add_column("metric", style="sek.key", no_wrap=True)
    t.add_column("role", no_wrap=True)
    t.add_column("why", ratio=1)
    role_style = {"primary": "sek.ok", "secondary": "sek.accent", "diagnostic": "sek.muted"}
    for m, role, why in r.items:
        t.add_row(f"{get_metric(m).display}\n[sek.muted]{m}[/]", f"[{role_style[role]}]{role}[/]", why)
    console.print(t)
    lab = lambda k, style="sek.muted": f"[{style}]{k:<13}[/]"  # noqa: E731
    extra = [Text.from_markup(lab("parameters") + json.dumps(r.params)),
             Text.from_markup(lab("empty policy") + r.empty_policy)]
    extra += [Text.from_markup(lab("statistics") + s) for s in r.statistics]
    extra += [Text.from_markup(lab("note", "sek.warn") + n) for n in r.notes]
    console.print(panel(Group(*extra), "how to report"))
    return 0


def cmd_datasets(a) -> int:
    from rich.table import Table

    from ._console import P, banner, console, panel
    from .datasets import DATASETS, get_dataset

    if a.name:
        d = get_dataset(a.name)
        console.print(panel(d.describe(), d.title))
        return 0
    banner(f"{len(DATASETS)} dataset presets", compact=True)
    t = Table(border_style=P["800"], header_style=f"bold {P['200']}")
    for c in ("key", "modality", "dataset", "official metrics"):
        t.add_column(c, style="sek.key" if c == "key" else None)
    for key, d in DATASETS.items():
        t.add_row(key, d.modality, d.title, ", ".join(d.metrics[:4]) + ("…" if len(d.metrics) > 4 else ""))
    console.print(t)
    return 0


def cmd_visualize(a) -> int:
    import numpy as np

    from . import viz
    from .io import load_volume

    ref_v = load_volume(a.ref)
    pred_v = load_volume(a.pred) if a.pred else None
    img = load_volume(a.image, kind="image").data if a.image else None
    val = [int(x) for x in a.label.split("+")] if a.label else None
    r = np.isin(ref_v.data, val) if val else ref_v.data != 0
    p = None if pred_v is None else (np.isin(pred_v.data, val) if val else pred_v.data != 0)
    window = a.window if a.window != "auto" else None
    if a.kind == "triplanar":
        fig = viz.triplanar(img, p, r, affine=ref_v.affine, window=window, title=a.title)
    elif a.kind == "montage":
        fig = viz.slice_montage(img, p, r, affine=ref_v.affine, window=window, title=a.title)
    elif a.kind == "projection":
        fig = viz.error_projection(p, r, affine=ref_v.affine, title=a.title)
    elif a.kind == "surface":
        if a.out.endswith(".html"):
            viz.surface_distance_map(p, r, ref_v.spacing, backend="plotly", html_path=a.out, title=a.title)
            print(a.out)
            return 0
        fig = viz.surface_distance_map(p, r, ref_v.spacing, title=a.title)
    else:
        fig = viz.error_overlay(img, p, r, affine=ref_v.affine, window=window, view=a.view, title=a.title)
    fig.savefig(a.out)
    from ._console import console

    console.print(f"[sek.ok]✓[/] [sek.key]{a.out}[/]")
    return 0


def cmd_home(a) -> int:
    from rich.table import Table

    from ._console import P, banner, console
    from .metrics import list_metrics

    banner(f"{len(list_metrics())} metrics · GPU surface distances · statistics · plots · reports")
    t = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    t.add_column(style="sek.key", no_wrap=True)
    t.add_column(style="grey78")
    for cmd, desc in (("evaluate", "score a prediction folder against a reference folder"),
                      ("report", "self-contained HTML report from a results folder"),
                      ("compare", "paired statistical comparison of two results folders"),
                      ("rank", "rank methods with bootstrap stability"),
                      ("recommend", "which metrics to report for your problem"),
                      ("visualize", "error overlays, projections, 3D surface-distance maps"),
                      ("metrics", "list every metric with direction and unit"),
                      ("datasets", "benchmark presets with official protocols")):
        t.add_row(f"segevalkit {cmd}", desc)
    console.print(t)
    console.print(f"\n[sek.muted]docs[/] [link=https://aj-das-research.github.io/SegEvalKit]"
                  f"[{P['300']}]https://aj-das-research.github.io/SegEvalKit[/][/link]   "
                  "[sek.muted]help[/] segevalkit <command> -h")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    try:
        from rich_argparse import RichHelpFormatter

        RichHelpFormatter.styles.update({
            "argparse.args": "bold #b89cf5", "argparse.groups": "bold #9a72ee", "argparse.metavar": "#1fb3a1",
            "argparse.prog": "bold #d6c7fb", "argparse.help": "default", "argparse.text": "default",
        })
        fmt_cls = RichHelpFormatter
    except ImportError:  # pragma: no cover
        fmt_cls = argparse.HelpFormatter
    p = argparse.ArgumentParser(prog="segevalkit", formatter_class=fmt_cls,
                                description="Evaluate volumetric medical image segmentation.")
    p.add_argument("--version", action="version", version=f"segevalkit {__version__}")
    p.set_defaults(fn=cmd_home)
    sub = p.add_subparsers(dest="cmd")

    e = sub.add_parser("evaluate", formatter_class=fmt_cls, help="evaluate a prediction folder against a reference folder")
    e.add_argument("--pred", help="prediction folder")
    e.add_argument("--ref", help="reference folder")
    e.add_argument("--prob", help="folder of per-structure probability maps (<case>/<label>.nii.gz)")
    e.add_argument("--labels", help="'liver=1,tumour=2' / 'kidney=2+3' / 'liver,pancreas' / JSON-YAML file")
    e.add_argument("--metrics", help="comma-separated metric names or sets (default, distance, detection, all...)")
    e.add_argument("--config", help="YAML/JSON configuration file")
    e.add_argument("--dataset", help="dataset preset (see `segevalkit datasets`)")
    e.add_argument("--out", help="output folder")
    e.add_argument("--name", help="method name stored in the results")
    e.add_argument("--workers", type=int, help="parallel worker processes")
    e.add_argument("--device", help="cpu | cuda | cuda:N")
    e.add_argument("--nsd-tolerance", type=float, help="NSD tolerance in mm")
    e.add_argument("--empty-policy", help="segevalkit | brats2023 | metrics_reloaded | nan")
    e.add_argument("--alignment", choices=["strict", "resample", "ignore"])
    e.add_argument("--connectivity", type=int, default=26, choices=[6, 18, 26])
    e.add_argument("--min-lesion-voxels", type=int, default=0)
    e.add_argument("--cases", help="text file with case ids to evaluate")
    e.add_argument("--report", action="store_true", help="also write report.html")
    e.add_argument("--images", help="image folder for report overlays")
    e.set_defaults(fn=cmd_evaluate)

    r = sub.add_parser("report", formatter_class=fmt_cls, help="build an HTML report from a results folder")
    r.add_argument("results")
    r.add_argument("--out")
    r.add_argument("--images")
    r.add_argument("--gallery-metric", default="dice")
    r.add_argument("--window", default="abdomen")
    r.set_defaults(fn=cmd_report)

    c = sub.add_parser("compare", formatter_class=fmt_cls, help="paired statistical comparison of two results folders")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--metrics")
    c.add_argument("--test", default="wilcoxon", choices=["wilcoxon", "ttest", "permutation"])
    c.add_argument("--correction", default="holm", choices=["holm", "bh", "none"])
    c.add_argument("--name-a")
    c.add_argument("--name-b")
    c.add_argument("--out")
    c.set_defaults(fn=cmd_compare)

    k = sub.add_parser("rank", formatter_class=fmt_cls, help="rank several results folders with bootstrap stability")
    k.add_argument("results", nargs="+")
    k.add_argument("--metric", default="dice")
    k.add_argument("--label")
    k.add_argument("--n-boot", type=int, default=1000)
    k.add_argument("--out", help="blob-plot PNG path")
    k.set_defaults(fn=cmd_rank)

    m = sub.add_parser("metrics", formatter_class=fmt_cls, help="list available metrics")
    m.add_argument("--family")
    m.add_argument("-v", "--verbose", action="store_true")
    m.set_defaults(fn=cmd_metrics)

    g = sub.add_parser("recommend", formatter_class=fmt_cls, help="recommend metrics for a problem fingerprint")
    g.add_argument("--structure", default="large_organ",
                   choices=["large_organ", "small_structure", "small_lesion", "large_lesion", "tubular", "hollow"])
    g.add_argument("--multi-instance", action="store_true")
    g.add_argument("--boundary-critical", action="store_true")
    g.add_argument("--volumetry", action="store_true")
    g.add_argument("--empty-references", action="store_true")
    g.add_argument("--probabilistic", action="store_true")
    g.add_argument("--asymmetric", action="store_true", help="FP and FN have different costs")
    g.add_argument("--noisy-reference", action="store_true")
    g.add_argument("--ranking", action="store_true")
    g.add_argument("--tolerance", type=float, help="acceptable boundary deviation in mm")
    g.add_argument("--markdown", action="store_true")
    g.set_defaults(fn=cmd_recommend)

    d = sub.add_parser("datasets", formatter_class=fmt_cls, help="list dataset presets or show one")
    d.add_argument("name", nargs="?")
    d.set_defaults(fn=cmd_datasets)

    v = sub.add_parser("visualize", formatter_class=fmt_cls, help="render a qualitative error figure for one case")
    v.add_argument("--pred")
    v.add_argument("--ref", required=True)
    v.add_argument("--image")
    v.add_argument("--label", help="label id(s), e.g. 1 or 2+3; default: any non-zero")
    v.add_argument("--kind", default="triplanar", choices=["slice", "triplanar", "montage", "projection", "surface"])
    v.add_argument("--view", default="axial", choices=["axial", "coronal", "sagittal"])
    v.add_argument("--window", default="abdomen")
    v.add_argument("--title")
    v.add_argument("--out", required=True)
    v.set_defaults(fn=cmd_visualize)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
