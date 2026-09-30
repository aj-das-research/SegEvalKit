# Choosing metrics

There is no single best segmentation metric. Each metric answers one question and is blind to others: Dice does
not see a distant false-positive island, Hausdorff distance does not see a thin missing slab, neither sees a vessel
cut in two, and none of them sees whether the probabilities can be trusted. Choosing metrics means choosing the
questions you need answered.

<div class="grid cards" markdown>

-   :material-compass-outline:{ .lg } **[Decision guide](choosing.md)**

    From the properties of your problem (structure size and shape, instances, boundary criticality, volumetry,
    probabilities, empty cases) to a justified metric set, with the recommender.

-   :material-alert-outline:{ .lg } **[Pitfalls](pitfalls.md)**

    The ways metrics mislead in 3D, each with a small numeric example and the fix.

-   :material-scale-balance:{ .lg } **[Conventions & conformance](conventions.md)**

    Why the same metric gives different numbers in different tools, which convention SegEvalKit uses, and the
    tests that pin it down.

-   :material-chart-bell-curve-cumulative:{ .lg } **[Sensitivity study](sensitivity-study.md)**

    What each metric actually notices, measured on real CT anatomy under controlled errors.

</div>

!!! note "The one-line rule"
    Report **one overlap metric and one boundary metric** for every structure, then add detection metrics when
    objects are instances, topology metrics when connectivity matters, volume agreement when volume is the
    endpoint, and calibration when probabilities are used (Maier-Hein et al. 2024, *Metrics Reloaded*).
