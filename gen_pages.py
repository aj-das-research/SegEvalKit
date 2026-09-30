"""Generate registry-driven documentation pages at build time (mkdocs-gen-files).

* ``metrics/catalogue.md``: every registered metric with its metadata, so the
  catalogue can never drift from the code.
* ``datasets/presets.md``: every dataset preset.
"""

import html
import math
import sys
from pathlib import Path

import mkdocs_gen_files

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segevalkit.datasets import DATASETS  # noqa: E402
from segevalkit.metrics import FAMILIES, METRIC_SETS, list_metrics  # noqa: E402

FAMILY_PAGE = {"overlap": "overlap", "volume": "volume", "distance": "distance", "topology": "topology",
               "detection": "detection", "calibration": "calibration", "agreement": "agreement"}
BETTER = {"higher": "↑ higher", "lower": "↓ lower", "zero": "→ 0", "none": "descriptive"}


def _prose(text):
    """Registry prose for Markdown: typographic relations and units, literal angle brackets."""
    for a, b in (("<=", "≤"), (">=", "≥"), ("mm^3", "mm³"), ("mm^2", "mm²")):
        text = text.replace(a, b)
    return html.escape(text, quote=False)


def _rng(r):
    f = lambda v: "∞" if math.isinf(v) and v > 0 else ("−∞" if math.isinf(v) else f"{v:g}".replace("-", "−"))  # noqa: E731
    lo = "(" if math.isinf(r[0]) else "["
    hi = ")" if math.isinf(r[1]) else "]"
    return f"{lo}{f(r[0])}, {f(r[1])}{hi}"


with mkdocs_gen_files.open("metrics/catalogue.md", "w") as fh:
    ms = list_metrics()
    fh.write("# Full metric catalogue\n\n")
    fh.write(f"SegEvalKit registers **{len(ms)} metrics** in {len(FAMILIES)} families. This page is generated from "
             "the metric registry at build time, so it always matches the installed code. Click a name for the "
             "full definition.\n\n")
    fh.write("```console\n$ segevalkit metrics -v      # the same list on the command line\n```\n\n")
    for fam, title in FAMILIES.items():
        rows = [m for m in ms if m.family == fam]
        if not rows:
            continue
        fh.write(f"## {title}\n\n<div class=\"sek-catalogue\" markdown>\n\n"
                 "| Metric and key | Better | Range | Unit | Needs | In one sentence |\n"
                 "|---|---|---|---|---|---|\n")
        for m in rows:
            needs = ", ".join(m.requires) or "–"
            fh.write(f"| [{m.display}]({FAMILY_PAGE[fam]}.md#{m.name})<br>`{m.name}` | {BETTER[m.better]} | "
                     f"{_rng(m.value_range)} | {m.unit or '–'} | {needs} | {m.summary} |\n")
        fh.write("\n</div>\n\n")
    fh.write("## Metric sets\n\nNamed bundles accepted anywhere a metric list is expected "
             "(`metrics=[\"default\", \"cldice\"]`, `--metrics distance,detection`).\n\n| Set | Metrics |\n|---|---|\n")
    for k, v in METRIC_SETS.items():
        fh.write(f"| `{k}` | {', '.join(f'`{x}`' for x in v)} |\n")
    fh.write("| `all` | every registered metric |\n| `all_binary` | every metric that does not need probabilities |\n")

with mkdocs_gen_files.open("datasets/presets.md", "w") as fh:
    fh.write("# Dataset presets\n\nA preset stores a benchmark's label ids and regions, the layout of its reference "
             "labels, its official metrics and their parameters, and its empty-mask convention. Use it with\n\n"
             "```console\n$ segevalkit datasets               # list\n$ segevalkit datasets kits23        # details\n"
             "$ segevalkit evaluate --dataset kits23 --pred preds/ --ref kits23/ --out eval/\n```\n\n"
             "or in Python: `segevalkit.datasets.get_dataset(\"kits23\")`. This page is generated from the preset "
             "registry.\n\n")
    fh.write("| Key | Dataset | Modality | Cases | Official metrics |\n|---|---|---|---|---|\n")
    for k, d in DATASETS.items():
        fh.write(f"| [`{k}`](#{k}) | {d.title} | {d.modality} | {d.cases} | {', '.join(d.metrics[:5])}"
                 f"{'…' if len(d.metrics) > 5 else ''} |\n")
    for k, d in DATASETS.items():
        fh.write(f"\n## {d.title} {{#{k}}}\n\n")
        fh.write(f"<div class=\"sek-meta\"><span class=\"sek-chip\">{d.modality}</span>"
                 f"<span class=\"sek-chip teal\">{_prose(d.cases)}</span><span class=\"sek-chip orange\">{_prose(d.license)}</span></div>\n\n")
        fh.write(f"**Anatomy:** {_prose(d.anatomy)}  \n")
        fh.write(f"**Labels:** {d.label_summary()}  \n")
        lay = d.ref_layout or "auto"
        extra = d.ref_subdir or d.ref_file
        fh.write(f"**Reference layout:** `{lay}`" + (f" (`{extra}`)" if extra else "") + "  \n")
        fh.write("**Metrics:** " + ", ".join(f"`{m}`" for m in d.metrics) + "  \n")
        if d.params:
            fh.write("**Parameters:** " + "; ".join(f"`{m}`: " + ", ".join(f"`{k}={v}`" for k, v in p.items()) for m, p in d.params.items()) + "  \n")
        fh.write(f"**Empty policy:** `{d.empty_policy}`  \n")
        fh.write(f"**Source:** <{d.url}>  \n**Cite:** {d.citation}\n")
        if d.notes:
            fh.write(f"\n!!! note \"Protocol notes\"\n    {_prose(d.notes)}\n")
