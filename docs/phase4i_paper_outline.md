# CVPR-style report outline

## Abstract

Study whether a single reconstruction trial can correct a false scene merge while preserving useful image measurements. On the Haiper design set a fixed split improved the combined metric by 0.164288 while mAA fell by 0.005556. A frozen automatic gate rejected that split on landmark count. Introduce the observation-indexed candidate and state explicitly that its prospective results are pending.

## Introduction

Motivate ambiguous matches between repeated structures and the cost of drawing conclusions from a correct-looking cluster or a leaderboard change. Ask whether a label-free decision made after one bounded trial improves grouping without losing registration or degrading pose accuracy.

## Related work

Cover normalized cuts, multi-stage/multi-model SfM, pose-consistency verification, loop constraints, repeated-feature visibility graphs, track communities and learned global SfM edge filtering. Compare intervention unit, training supervision, access to reconstructions and compute. See the architecture review for primary references.

## Problem formulation

Given an unordered image set and a frozen matching front end, return a scene assignment and world-to-camera pose or an outlier for each image. Treat independently reconstructed coordinate frames up to similarity. Scene labels and reference poses enter only the administrative scorer.

## Baselines

Report organizer full-development combined 0.418230 across 11 datasets. On the four reproduced ns64 development datasets, equal-dataset descriptive means were 0.525972 for ns64 and 0.353320 for the organizer. Mark coverage 203/1,563 and prior design exposure. Include same-parent pass-through, graph-only proposal, v2 feedback and paired-cost-only ablations.

## Proposed method and mathematical details

Describe connected normalized-cut proposals, one audited trial and unchanged mapper options. Give the identifiable keypoint-row residual definition, clipped paired difference, missing-observation penalty, per-image retention and upper-quartile safeguards from phase4i_mathematical_method.md. Show how gauge changes affect poses but preserve image-plane comparisons.

## Experimental protocol

State the frozen identities, no training, opaque IDs, deterministic input order, two prospective source groups, one trial, leakage boundary, resource caps and both evaluator variants. Explain score restoration after sealing, source-group dependence and operator-visible failure receipts.

## Results and ablations

| Historical Haiper experiment | mAA | Clusterness | Combined | Registered |
|---|---:|---:|---:|---:|
| ns64 pass-through | 0.744444 | 0.635294 | 0.685552 | 54/54 |
| Exploratory fixed split | 0.738889 | 1.000000 | 0.849840 | 54/54 |
| Frozen v2 selector, offline decision | 0.744444 | 0.635294 | 0.685552 | 54/54 |

The third row reuses identical selected prediction bytes. The trial split the 31-image model into 15 and 16 cameras; all 23 control poses were unchanged. These results come from one designed comparison. The final candidate has zero prospective paired scores. Populate its results table only after return validation, including both groups, every fallback, missing ablations, paired deltas, registration and resource denominators.

## Failure analysis and limitations

Explain the 63,829 / 73,841 reconstructed point veto and missing historical indexed exports. Distinguish registration from scene correctness, loss of genuine observations from repair of false tracks, and the effects of separate bundle adjustments. Discuss in-sample residual bias, coherent aliases, incorrect camera calibration, model flexibility, repeated-feature leakage, and runtime censoring. Free-runtime failures are operational evidence. They do not measure vision quality.

## Reproducibility

Publish source/configuration and archive digests, the verified dataset/split/evaluator identity, exact environment/weights, no training, proposal tie rules, output contract, simulation and tests, stage receipts and the scoring command. Keep dataset images, ground truth mappings and licensed weights outside the review ZIP.

## References

Shi and Malik, normalized cuts. Zach et al., visual loop constraints. Wilson and Snavely, repeated-structure visibility networks. Gong et al., cluster pose verification. TC-SfM, track-community reconstruction. Damblon et al., learned edge filtering. Cite the original ns64 implementation and COLMAP for the matching and mapping baselines. The architecture decision links the primary papers. A general superiority or publication-priority claim requires additional evidence.
