# Received Kaggle audit review

Reviewed 2026-11-11. Source: user-supplied root `imc_dataset_audit.zip`; reports preserved in `docs/evidence/kaggle_audit_received/`. All 12 manifest entries matched their recorded SHA256 and byte length; ZIP CRC check passed, with no omitted artifacts. Self-hashes verify transfer consistency, not independent authentication of Kaggle provenance. Reported mount is `/kaggle/input/competitions/image-matching-challenge-2025-ongoing`.

## Measured results from returned reports

- 2,036 mounted files, 5,218,961,459 bytes (4.861 GiB), complete scan, no archives found. This is mounted file size, not a compressed download size or hidden-test footprint.
- Training: 13 named datasets, 30 non-outlier scenes, 1,945 unique dataset/image records; 122 outliers (6.27%). Non-outlier scenes contain 9–200 images; datasets contain 22–225 images.
- Pose validation: 1,823 valid finite poses and 122 all-NaN poses; no duplicate composite keys, inconsistent assignments, malformed pose records or unresolved training references reported. Matching totals do not independently prove that every NaN row is an outlier: a row-level cross-tab was not exported.
- Visible test files: 73 images, comprising ETs (22) and stairs (51). These visible files do not establish hidden scoring-set size or disjointness from training.
- Outliers occur in only four named datasets: ETs 3, imc2023_heritage 61, imc2024_dioscuri_baalshamin 24, imc2024_lizard_pond 34. Maximum dataset outlier rate is 29.19%.
- Returned environment: Linux, Python 3.13.18, NumPy 2.1.3; successful write probe. This EDA environment is not automatically compatible with the earlier first-place pipeline requirements.

## Important contract correction

Inventory records sample_submission.csv (515,334 bytes) with SIX columns: `image_id,dataset,scene,image,rotation_matrix,translation_vector`. The training-label CSV has the previously documented five columns. Our notebook required exact five-column equality when selecting the sample and therefore reported it unavailable/ambiguous. This is a limitation of our parser, not proof the sample is absent.

The sample's row count, image_id semantics, uniqueness and visible-test coverage have NOT been validated. Preserve image_id and actual sample order in a future contract inspection; do not silently drop it based on historical prose. The size alone does not justify inferring a row count or hidden-test composition. `statistics_complete=true` currently means training parsing/file scanning completed, not that submission/evaluator review completed.

## Thresholds

train_thresholds.csv is present (1,806 bytes); SHA256 `5b92db83d510ab19e66bd1ef4c23865245d75b4a981986e6d8eb1aee06fc9fe2`. Thirty scene rows were exported. Twenty-seven passed the audit's positive-finite threshold check. Three—bike, chairs and fountain in imc2023_haiper—failed it, and raw values were replaced with null in the report. Therefore their actual values and failure reason are unknown. This may reflect legitimate format/convention or an audit assumption; do not classify them as corrupt or substitute synthetic thresholds. Obtain the raw small threshold CSV or a lossless follow-up report.

## Validation implications and recommendation

Do not freeze random 60/20/20 allocation. The current generator would allocate 9/2/2 datasets, which is approximately 69/15/15, not 60/20/20; two held-out groups can be unrepresentative given large size and outlier variation. Named datasets are not yet verified independent groups.

The scene name dioscuri appears in both imc2023_heritage and imc2024_dioscuri_baalshamin. This is a concrete provenance-overlap candidate, not proof of duplicate imagery. Audit exact/near duplicates and shared sources before assigning either group across partitions. Visible train/test ETs and stairs names also warrant overlap inspection; no test images should become tuning evidence.

After provenance grouping, consider fixed grouped development folds with a protected confirmation set if enough independent domains remain. Leave-one-group-out offers broader coverage but cost must be measured later. Balance outlier/domain coverage without looking at model scores. Preserve the same baseline, split, evaluator, thresholds and conditions for every improvement as required by AGENTS.md. No split or baseline is frozen by this review.

## Next bounded action

Inspect the actual sample submission and raw threshold metadata on Kaggle, preserving the extra identifier column and rejected raw threshold values. Add explicit sample/threshold completeness checks and per-scene NaN/outlier cross-tabs, then perform a source-overlap audit before split approval. No full image download, model run or submission is needed for the metadata correction. Official mAA-first versus participant overlap-first parity remains unresolved.

Readiness: real training-data evidence is now available; local API access is irrelevant to this successful notebook route. Baseline reproduction and split freezing still await contract/threshold reconciliation, source independence review and supervisor approval.
