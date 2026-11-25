# Phase 2D validation protocol

> Update after metadata receipt: **FROZEN**. All three original CSV hashes match the prior audit; full image membership and all hard constraints pass. The earlier draft/missing-input descriptions below are historical and superseded by `step02d_metadata_freeze_review.md`. No Phase 3 run is authorized by this freeze.


Status: DRAFT, NOT FROZEN. Version `imc2025_local_v1`, seed 20261111. No model scores were inspected. Original label and sample CSV bytes are missing locally; full image membership must not be inferred from filenames. Original threshold bytes also require export verification. The old `scripts/create_validation_splits.py` is exploratory and is superseded by this protocol; its random allocation is not a benchmark.

| Design | Scientific benefit | Cost and limitation | Decision |
|---|---|---|---|
| Grouped K-fold | All groups get held-out development predictions | Repeated tuning consumes every group; no independent confirmation | Useful development component only |
| Two grouped folds plus protected confirmation | Cross-group development plus a final untouched check | Only two confirmation groups; weak generalization evidence | Selected provisionally |
| Leave-one-group-out | Twelve separate group estimates | Twelve fitting/evaluation cycles per method; still little independent evidence and tuning leakage without lock | Too costly for primary protocol |
| Scene-disjoint diagnostic | Tests within-source scene transfer | Shared provenance and altered scene/outlier mixtures; not independent cross-domain evidence | Optional separately labelled future diagnostic, not generated |

| Evaluation partition | Groups | Datasets | Images | Non-outlier scenes | Outliers | Finite poses |
|---|---:|---:|---:|---:|---:|---:|
| Development A | 5 | 6 | 728 | 13 | 85 | 643 |
| Development B | 5 | 5 | 835 | 13 | 3 | 832 |
| Confirmation | 2 | 2 | 382 | 4 | 34 | 348 |
| Total | 12 | 13 | 1945 | 30 | 122 | 1823 |

These are proposed assignments using returned aggregate metadata, not validated image-level splits. Fold A trains/calibrates exclusively on B and evaluates A; fold B trains/calibrates exclusively on A and evaluates B. Confirmation is excluded from both. A selection procedure can combine development results only. Lock chosen configuration, checkpoint policy and code before a supervisor-authorized confirmation run; fit permitted learned components on all development data only. Do not repeatedly tune from confirmation results. Confirmation access is a governance boundary, not filesystem security; experiment validation requires a recorded confirmation release ID.

The merged group retains all 347 records, five scenes and 85 outliers. The allocation balances development scene counts and keeps small Haiper/stairs scenes in A and the 9/10-camera ET scenes in B. Reported non-outlier scene sizes range from 9 to 200. No scene is cut up to improve balance. The two confirmation groups provisionally cover lizard/pond and urban landmark mixtures and retain 34 outliers. Those semantic labels are weak proxies for domain diversity. Two groups are insufficient for a broad generalization claim. The 85-versus-3 development outlier imbalance cannot be interpreted as comparable fold difficulty; report per-dataset components and failures. Shared PhotoTourism or unknown pretraining provenance can still induce overlap beyond the eight detected pairs.

Primary aggregate: equal mean over the 11 development dataset scores from their held-out fold, not an unweighted mean of the two fold means (six versus five datasets). Report both fold means, all dataset scores and group-level sensitivity summaries; confirmation mean is reported separately. Cluster uncertainty resampling by candidate source group, never by image; with 10 development and two confirmation groups avoid strong significance claims. Compare paired differences, runtime, memory and failures on exactly matching splits. Registration coverage uses finite registered predictions divided by all input records, with a separately labelled non-outlier denominator diagnostic. Preserve failures in denominators.

Freeze gate: obtain byte-identical metadata; verify all image identities, 30-scene counts, 122 outliers, duplicate endpoints and dataset coverage; generate exclusive new membership artifact; review balance and provenance limitations; pin exact evaluator adapter, environment, split and threshold hashes; record supervisor acceptance. The current scripts never auto-freeze. Once frozen, corrections require a new benchmark version and reruns of baseline and every compared variant. Pretrained weights with unknown overlap must be declared; held-out images cannot train/calibrate any learned component.
