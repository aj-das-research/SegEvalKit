# About

SegEvalKit is an open-source library for the evaluation of volumetric medical image segmentation, developed by
Abhijit Das. It grew out of a practical problem: every segmentation project re-implements Dice and Hausdorff
distance slightly differently, rarely reports how empty masks were handled, and rarely goes beyond a mean Dice.
SegEvalKit collects the metrics the literature recommends, makes their conventions explicit and tested, and adds
the statistics and figures that turn per-case numbers into defensible conclusions.

- **Source:** <https://github.com/aj-das-research/SegEvalKit>
- **Licence:** Apache-2.0
- [Contributing](contributing.md) · [Changelog](changelog.md) · [Citation](citation.md)
