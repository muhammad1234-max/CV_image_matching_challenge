# Kaggle-native audit execution guide

Prepared 2026-11-11. Notebook: `notebooks/step02b_kaggle_dataset_audit.ipynb`. It embeds `scripts/kaggle_dataset_audit.py` and requires only Python standard libraries and NumPy. It has no API client, authentication, downloads, subprocess commands, package installation or network calls.

1. Open https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing in your browser and sign in yourself. Review/accept the competition rules if required for access.
2. Create a notebook associated with this ongoing competition, or open a new notebook and attach the competition through the input/data controls. Confirm that the attachment is the ongoing competition, not the original 2025 competition or an unrelated public copy.
3. Use the notebook editor's import/upload function to import the supplied `.ipynb`. The exact current menu labels could not be verified: https://www.kaggle.com/docs/notebooks returned no readable text. No invented button names are required for these actions.
4. Select CPU execution (no GPU/TPU accelerator), and disable internet. There are no secret/API credentials to configure.
5. Run all cells. Discovery searches beneath `/kaggle/input` for CSV pose-schema files alongside train/test directories, to a bounded depth. The mount name is not hard-coded. A unique match is selected; zero/multiple matches stop safely with a message.
6. If necessary, set `INPUT_PATH` to the actual attached root beneath `/kaggle/input`. Set `TRAIN_CSV` and `SAMPLE_CSV` to relative paths if metadata names are ambiguous or nonstandard. Select only actual training labels and the sample submission; never hidden-test labels. Do not substitute a guessed path. Archive-only layouts may require manual selection and will not yield image/pose statistics until metadata is available separately.
7. Inspect `/kaggle/working/imc_audit/audit_status.json`. `BLOCKED` requires resolving the reported issue. `AUDITED` means reports were produced, not that all semantics/provenance were validated. Examine completeness flags, unresolved references and source paths.
8. Download `/kaggle/working/imc_audit/imc_dataset_audit.zip` using the notebook output-file controls. If necessary, save a notebook version to retain outputs. Do not submit predictions or publish the notebook for this task. Return the ZIP for supervisor review.
9. If attachment or permission fails, check browser login, competition identity, rule acceptance and account access. Stop there; no local API login or credential sharing is required. A missing-input run still writes a small status ZIP.

## Outputs and bounds

Successful runs produce dataset_inventory.json, dataset_summary.json, dataset_groups.csv, scene_statistics.csv, outlier_statistics.csv, pose_validation.json, submission_contract.json, validation_recommendation.md, execution_environment.json, audit_status.json, threshold_provenance.json and scene_sizes.svg. The ZIP also includes archive_manifest.json with report hashes and any omissions. Blocked runs produce only artifacts available before the blocker; do not mistake stale files outside the ZIP from a prior run for new results.

One selected-root inventory scan indexes paths on temporary SQLite storage. It reads no image pixels. Limits: one million scanned files/CSV rows, 20,000 exported inventory entries, 1,000 cataloged metadata/archive entries, hashing of the first 200 eligible metadata files up to 10 MiB each. ZIP member inspection is limited to ZIP files <=64 MiB (central directory only, first 100 member names exported); larger ZIPs and other archives are recorded but not opened. No archives are extracted. Threshold export is capped at 10,000 rows per eligible file; exact units are not inferred. Catalog/count completeness flags must be checked.

The output ZIP permits only explicitly generated reports; excludes source images, source CSV copies, raw test rows, credentials and SQLite files. Per-report limit is 10 MiB; overall uncompressed report budget 25 MiB. Relative image paths may occur in the bounded inventory, but test-image contents and labels are not exported. The archive manifest records omitted reports if limits are reached.

The notebook probes output-directory writability at runtime. Local fixtures verified that path handling works, but actual Kaggle writability/permissions remain untested until you execute it. No Windows paths occur in executable notebook code.
