# Returned Phase 2C audit review

Reviewed 2026-11-11. Source: root imc_contract_audit.zip; preserved reports in docs/evidence/contract_audit_received. ZIP CRC and all 13 manifest entries' byte sizes/SHA256 passed. No omitted reports. Hashes establish transfer consistency, not independent certification of source provenance.

## Submission contract

Actual sample: 1,945 rows, 1,945 unique image_id values and 1,945 unique (dataset,image) keys, with no missing/duplicate identifier issues. Six-column order is image_id,dataset,scene,image,rotation_matrix,translation_vector. There are 1,937 distinct image strings and eight basenames repeated across datasets, supporting composite-key handling. All scene values are cluster0 and all pose fields parse as numeric vectors; these are sample patterns, not ground-truth predictions.

All 73 visible test keys occur in the sample; 1,872 sample rows have no matching visible test file. This does not establish hidden scoring-set size or authorize dropping sample rows. Sample counts by dataset match the previously observed training counts, but aggregate agreement is not proof of row-level identity. Preserve original identifiers/order and inspect the runtime sample when eventually submitting. Final scorer acceptance remains unverified.

## Threshold reconciliation

The source threshold hash matches the previous audit. The three Haiper scenes (bike, chairs, fountain) each contain exactly:

`0;0;0.001;0.025;0.05;0.1`

The previous validator rejected the first two zero entries, not malformed numeric encoding. The saved participant reference's tth_from_csv accepts semicolon-separated float values; mAA_on_cameras skips the first two thresholds by default. Thus these zeros are compatible with that reference's scoring configuration, whose retained Haiper scoring thresholds are 0.001, 0.025, 0.05 and 0.1. Preserve all six source values; do not globally delete zeros or invent replacements. This resolves the cause of rejection and reference-code compatibility, not official intent, units or deployed-scorer parity. Source: docs/evidence/public_sources/first_metric.py.txt and returned thresholds_raw_and_parsed.csv.

## Pose/outlier integrity

The row-level audit reports nonoutlier_all_nan=0 and outlier_not_all_nan=0. Together with the returned contingency table, all 122 all-NaN records correspond to outliers; the 1,823 other rows are finite. No finite rotation failed the stated orthogonality absolute tolerance 0.001 (relative tolerance zero) or determinant tolerance 0.001. No ground-truth records were altered.

## Overlap remains incomplete

Inventory: 1,945 training images. Exact hashes completed: 8/16 same-size cross-dataset candidates. Thumbnail hashes: 738/1,945. Cross-dataset pair comparisons: 0/1,713,342. The 120-second budget expired during image processing. Therefore the empty candidate table and singleton groups provide NO evidence that datasets are independent or duplicate-free.

Next bounded action: in the same Kaggle session/output directory, set MAX_OVERLAP_SECONDS=900 and rerun the execution cell. Keep CPU and internet off. The cache reuses completed hashes only when its catalog matches. Nine hundred seconds is a suggested budget, not a completion guarantee. If the session reset, cached work may need repeating. Return the refreshed ZIP and inspect overlap_complete/overlap_status.json before drawing conclusions. Exact byte matches require grouping; perceptual candidates still need conservative manual review.

## Readiness

Submission schema and pose/outlier consistency are now evidenced. Haiper rejection is explained for the pinned participant reference. Source independence, completed duplicate screening, manual near-match review and official evaluator parity remain unresolved. Keep the dioscuri pair conservatively together pending review. No split frozen, model run or prediction submission authorized by this review. All eventual improvements retain the same frozen baseline and evaluation conditions.
