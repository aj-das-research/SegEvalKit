# Developer guide

This part of the documentation is for people who **extend, maintain or embed** SegEvalKit: adding a metric,
supporting a new dataset, adding a plot, keeping conventions honest, and shipping a release. It assumes you have
read the [Quickstart](../getting-started/quickstart.md).

<div class="grid cards" markdown>

-   **[Architecture](architecture.md)**

    The layers of the library, the life of one evaluation, and the design rules that keep it correct.

-   **[Adding a metric](adding-a-metric.md)**

    A complete worked example: implement, register, cache, handle empty masks, test, document.

-   **[Adding a dataset preset](adding-a-dataset.md)**

    Encode a benchmark's labels, layout and official protocol so anyone can reproduce it with one flag.

-   **[Plots, themes & visualisation](plots-and-themes.md)**

    The purple design system, colour-vision checks, and how to add a plot that fits.

-   **[Testing & conformance](testing.md)**

    Analytic tests, cross-library conformance, backend equivalence, end-to-end and CLI tests.

-   **[Documentation](documentation.md)**

    How this site is built, generated pages, writing style, and deployment to GitHub Pages.

-   **[Releasing & maintenance](releasing.md)**

    Versioning, the results-format contract, deprecation policy, CI and the release checklist.

</div>
