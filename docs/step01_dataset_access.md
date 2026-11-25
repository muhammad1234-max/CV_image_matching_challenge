# Step 1 — Dataset access

Audit date: **2026-11-10, Asia/Karachi**.

## Actual authentication result

`python scripts/audit_environment.py` detected the existing user Kaggle credential file, with no credential environment variables set. Only presence booleans were logged. The installed Kaggle API is 2.2.4, although `kaggle` is not on PATH. `python scripts/audit_kaggle.py` loaded authentication and made a read-only file-list request for `image-matching-challenge-2025-ongoing`.

Result: **HTTP 401 Unauthorized; zero files listed; no downloads.** See [sanitized API result](evidence/kaggle_access.json). Credential loading is not successful authentication. The exact account-side cause is unknown; expired, invalid or otherwise unusable authentication is possible. No second request was made after this failure. Error bodies/headers/signed URLs were suppressed to avoid secret leakage.

Rules acceptance remains **unknown**, rather than inferred false from a 401. No rules were accepted on the user's behalf. No data archives, metadata CSVs, model weights or submissions were transferred. The small downloaded GitHub text files are research references, not competition data.

## Minimal manual action

Sign into Kaggle, review/join the [ongoing competition and its rules](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/rules), and refresh the local API authentication using [Kaggle's authentication documentation](https://github.com/Kaggle/kaggle-cli). Keep secrets out of chat and the repository. Once repaired, a later authorized step can rerun the bounded audit. No credentials need to be shared with the supervisor.

## Sizes and layout

| Quantity | Status |
|---|---|
| Current file names, count and individual sizes | Unknown: listing failed |
| Archive compressed size | Unknown; cannot derive it from historical layouts |
| Decompressed total / hidden-test size | Unknown |
| Ongoing sample-submission content | Not downloaded |
| Local D: free | 142,126,239,744 bytes = 132.37 GiB, measured by `shutil.disk_usage` |
| Local C: free | 2.43 GiB, same audit; avoid caches here |

Historical layout documented in the [first-place README](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025): `sample_submission.csv`, `train_labels.csv`, `train_thresholds.csv`, `train/<dataset>/`, and `test/<dataset>/`. Examples include ETs and stairs; these are not a verified ongoing inventory. Training includes scene folders/outliers, while test images are mixed within a dataset according to the [official task description](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/overview/evaluation).

## Proposed download and integrity strategy — not executed

1. Establish authenticated listing and save all pages, exact byte sizes and retrieval date. Listing alone does not prove download permission or rules acceptance.
2. Retrieve only `sample_submission.csv`, `train_labels.csv`, `train_thresholds.csv` first, each capped at 5 MB in the existing audit script. Confirm the manifest names before assuming they exist.
3. Estimate storage from actual metadata: if `A` is downloaded archive bytes and `U` is uncompressed bytes, reserve at least `A + U + weights + feature/match caches + reconstruction output + working headroom`. Proposed headroom is 20% above this sum; this is a planning policy, not a dataset-size estimate. If `U` is unavailable, inspect an archive directory or a representative authorized subset before committing to a full download.
4. In a later approved data-acquisition step, download to `D:\CV_Project\data\raw` or chosen Linux storage, one bounded subset/archive at a time. Keep the original archive until extraction is verified. Do not use `download all` during Step 1.
5. Record local SHA256 and byte length for every acquired artifact; compare publisher hashes when provided. A self-computed hash supports repeatability but does not prove authenticity. Verify ZIP CRCs, safe extraction paths, counts and CSV-to-image references; check duplicate IDs, unreadable images and metadata/schema consistency.
6. Keep dataset and generated artifacts ignored; retain only sanitized manifests and validation summaries in version control. Preserve source/license provenance.

No numerical decompression ratio or download-duration estimate is defensible without the manifest and measured bandwidth. The optional NVS-NAR-IQA dataset is outside this step and was not acquired.

## Step 1.5 review addendum

Original findings above are retained as the Step 1 record. The later [supervisor review](step015_supervisor_review.md) qualifies CLI availability/authentication, evaluator validation and current rule uncertainty. The [execution plan](step015_execution_environment.md) supersedes the initial baseline order with starter-first integration, followed by a verified top-10 target. Current readiness: [BLOCKED](step015_step2_readiness.md).
