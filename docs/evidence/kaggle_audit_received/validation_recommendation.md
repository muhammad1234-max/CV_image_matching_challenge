# Validation review
Observed labeled dataset names: 13. Independent source groups: UNKNOWN.
The 60/20/20 allocation is NOT approved or frozen. With 13 named groups, each 20% partition would contain approximately 2 groups; inspect dataset_groups.csv, outlier_statistics.csv and scene_statistics.csv for imbalance.
Before assignment, audit source provenance, exact duplicates and near-duplicates across groups. No image-content duplicate scan was run. Names and sizes cannot establish independence.
If few independent groups remain, consider leave-one-independent-group-out development CV, with a separately protected confirmation set if feasible. Its compute cost needs a later baseline pilot; feasibility is not established by this audit. Scene-disjoint diagnostics can complement dataset-level generalization but cannot replace it. Small scenes remain reported; do not exclude them post hoc.
Freeze baseline, data, evaluator, thresholds and conditions once approved; every improvement uses that same version. Never tune on confirmation. Official prose mAA-first assignment remains distinct from participant overlap-first code; no parity claim.
