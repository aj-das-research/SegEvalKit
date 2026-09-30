# Documentation

This site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and published to GitHub Pages.

```console
$ pip install -e ".[docs]"
$ mkdocs serve                 # live preview at http://127.0.0.1:8000
$ mkdocs build --strict        # what CI runs: warnings are errors
$ mkdocs gh-deploy             # publish to the gh-pages branch
```

## Structure

| Source | Content |
|---|---|
| `docs/*.md` | hand-written pages |
| `docs/gen_pages.py` | generates `metrics/catalogue.md` and `datasets/presets.md` from the registries at build time |
| `docs/api/*.md` | `::: module` directives rendered by mkdocstrings from the docstrings |
| `research/*.md` | the literature review and dataset survey, included verbatim with snippets |
| `docs/assets/figures/` | figures rendered by `benchmarks/*` scripts from real results |
| `docs/stylesheets/segevalkit.css` | the purple theme and the academic serif typography (STIX Two Text, a Times-family face in the spirit of the ICLR template) |

## Writing style

* Lead with the question a page answers; keep the first paragraph short.
* Every metric statement matches the implementation. If code and docs disagree, fix one of them in the same change.
* Equations use `\[ ... \]` (display) and `\( ... \)` (inline).
* Every number shown in the docs comes from a script in `benchmarks/`; regenerate figures rather than editing them.
* Prefer tables for comparisons and admonitions only for things a reader must not miss.
