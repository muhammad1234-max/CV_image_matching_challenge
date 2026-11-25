# Validation review pending Kaggle evidence

The requested Kaggle-native notebook is prepared; no real dataset statistics were obtained locally. Dataset names, scenes per group, source diversity, scene-size variability, outlier percentages and source overlap still require the actual audit outputs. Do not freeze the earlier 60/20/20 proposal.

The notebook reports observed dataset-label groups, their unique image counts, scene counts, outlier percentages and scene-size distribution. These support representativeness review, not independence certification. It does not hash image pixels or detect near-duplicates; image-provenance and duplicate-source checks remain a separate gate. Repeated basenames across different datasets are permitted; duplicate identity checks use `(dataset,image)`.

After execution, examine whether a 20% partition would contain too few groups or omit important scene types/outlier regimes. Source-related datasets must be grouped before splitting. If independent groups are few, assess leave-one-independent-group-out development CV; compute cost remains unknown until a later authorized baseline pilot. Preserve a confirmation set where defensible; never tune hyperparameters on it. A separately frozen scene-disjoint diagnostic may test within-domain behavior but does not replace dataset-level generalization. Tiny scenes and reconstruction failures must remain visible, not excluded after scores are observed.

All future improvements must use the same frozen baseline, data/split, evaluator, threshold source and experimental conditions, as required by AGENTS.md. Any correction changes the benchmark version and requires rerunning the baseline and all compared methods. No actual split is created by this notebook.

Official prose's mAA-first assignment remains distinct from the participant implementation's overlap-first assignment. Reference code is not proof of leaderboard parity. Threshold CSVs, when found by schema, are recorded with source path/hash and positive-finite-value validation; no synthetic values are substituted. Their official provenance must be confirmed from the attached competition. Sources: https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/overview/evaluation and the exact participant reference recorded in step015_metric_reconciliation.md.

Current decision: wait for the real ZIP and source-overlap assessment. Neither 60/20/20 nor leave-one-group-out is approved for experiments yet.
