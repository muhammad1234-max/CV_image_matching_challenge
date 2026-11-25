# Metadata receipt and benchmark freeze review

The returned `imc_benchmark_metadata.zip` was inspected before extraction: exactly four expected files, less than 10 MiB uncompressed, no arbitrary extraction paths. All three source CSV SHA256 values match the earlier audit and returned manifest. Original bytes are preserved in `docs/evidence/benchmark_metadata_received/`.

Archive SHA256: `24ab3a4fb90858be2a9f6e24f1a6a5d915000812379787ab6a2e257a1030b894`.

## Final status

`imc2025_local_v1` is **FROZEN** under the existing Prompt 05 authorization to freeze after evidence and constraints pass. This is the first freeze; prior draft manifests are preserved in the received evidence directory. `benchmark.json` and `image_membership.json` are authoritative. Historical draft proposal/source-availability fields remain preserved as provenance; `freeze_decision.json`, `metadata_validation.json` and the final benchmark status supersede those fields.

- benchmark_hash: `a78f0542413b8ac2ec8c262988ed5303c934490a2b31d2a45d3939dd6d3ab2a4`
- split_hash: `170546743202f21b9bcc98376ff09a6eae2da16ce833115478a976c4f46b2057`
- dataset_inventory_hash: `ad6c64f09b9590cd2ab1c67a7c472ce52aa761ab5c5297f30884faa9ad85ad37`
- evaluator_hash: `c7ae94306ba0ae4ce9c00e9abc7e15bd99f3091fa063b5826b0dd567a6c3f261`
- threshold_hash: `c6aec2fc2c8f9cb5f268f57333911f0835fcaba8aedc0eec67951ed729e68645`

## Verified results

| Partition | Candidate groups | Datasets | Images | Non-outlier scenes | Outliers |
|---|---:|---:|---:|---:|---:|
| Development A | 5 | 6 | 728 | 13 | 85 |
| Development B | 5 | 5 | 835 | 13 | 3 |
| Confirmation | 2 | 2 | 382 | 4 | 34 |
| Total | 12 | 13 | 1945 | 30 | 122 |

The allocation is unchanged. Every original image has a unique dataset/image identity and explicit evaluation/training/excluded-confirmation roles. All identities occur in the returned image inventory. All 122 NaN poses are exactly outlier records; all other 1823 poses are finite and pass rotation checks. All eight reported duplicate endpoints exist and remain together in the merged heritage/Dioscuri group. All 30 original threshold strings exactly match the pinned decoded strings, including six zero values. Original sample order and all 1945 unique identities are preserved and fingerprinted.

The real sample exposed a draft validator error: sample pose matrices are arbitrary placeholders, not physical rotations. The checker now exempts only the sample template from rotation geometry checks; finite dimensions and unique identities are still checked. Ground-truth and prediction rotations remain strict. A dedicated regression covers this distinction. The correction happened before any freeze or model comparison; no baseline rerun was needed.

The complete lightweight suite passed before freeze (57 tests, 3.978 seconds). Post-freeze integrity and full suite were rerun successfully; see the accompanying test receipt. New tests include authentic metadata membership, sample placeholders and refusing to overwrite a frozen benchmark. `python scripts/validate_benchmark.py` now rechecks original metadata, file/code hashes, all membership constraints and frozen split contents.

## Interpretation and next phase

The freeze establishes a fixed local comparison contract, not guaranteed source independence or hidden Kaggle parity. Image bytes have not been independently rehashed locally; duplicate and perceptual screening conclusions still rely on the returned audit. Two confirmation groups and development outlier imbalance limit statistical claims. Source commit is unavailable; evaluator bytes and adapter hashes are pinned. CPU evaluator environment is recorded; GPU baseline code/config/weights/environment remain unregistered.

No additional metadata export is required. Next human action is supervisor approval for Phase 3. Before its first GPU run, select and pin the baseline implementation, weights, configuration, full source snapshot and GPU environment. This workspace has no Git repository, so do not invent a Git commit; record that status and an exact source snapshot or initialize version control when authorized. The proposed ETs smoke test remains a wiring check, followed by both full development folds for scientific baseline registration. Confirmation remains excluded from model selection.

No model training, SfM, large image download or Kaggle submission was performed. STOP after Phase 2D.
