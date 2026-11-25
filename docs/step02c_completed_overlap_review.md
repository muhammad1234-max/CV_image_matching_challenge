# Completed overlap audit review

Reviewed 2026-11-11. Source: updated root imc_contract_audit.zip; preserved separately in docs/evidence/contract_audit_rerun_received. Prior run remains in contract_audit_received. All 13 manifest artifact hashes/lengths and ZIP CRC passed; no omitted artifacts. Integrity verification establishes consistency of the supplied reports, not an independent rehash of remotely mounted images.

## Completed coverage

The returned audit reports all 1,945 training images inventoried and thumbnail-hashed; all 16 same-size cross-dataset exact-hash candidates processed; all 1,713,342 cross-dataset pairs compared. No decoding errors. Complete=true, using 129.04 seconds of the 900-second budget. The inventory catalog hash matches the earlier run. Runtime may include reused cached work and is not a cold-start benchmark.

## Confirmed leakage constraint

The report identifies eight byte-identical image pairs between imc2023_heritage and imc2024_dioscuri_baalshamin. These datasets MUST remain in the same source group for any training/development/confirmation partition. Previously this was a precaution based on shared scene names; it is now supported by the exact-hash audit.

Affected basenames: dioscuri_archive_0003.png, dioscuri_archive_0004.png, dioscuri_archive_0010.png, dioscuri_archive_0015.png, dioscuri_archive_0026.png, dioscuri_archive_0230.png, dioscuri_img_0055.png, dioscuri_img_0105.png. Full paired paths are retained in dataset_overlap_candidates.csv.

Merging this pair produces 12 candidate grouping components from 13 named datasets. They are not 12 proven independent sources. The combined component contains 347 image records and 85 outlier records based on the earlier dataset counts; no deduplication or removal has been performed. Keeping them together avoids these eight known duplicates crossing partitions without changing the official dataset composition.

No additional perceptual candidates were reported at the configured 64-bit dHash distance <=5. This screening can miss crops, substantial viewpoint changes or other shared-source imagery. Exact matches take precedence in candidate classification; no separate perceptual row is expected for those eight pairs. The generic NEAR_MATCHES_UNREVIEWED flag remains in status, but the candidate CSV contains no non-exact perceptual matches requiring review. Absence of candidates does not certify independence.

## Other findings retained

Submission, threshold and pose reports are byte-identical to the prior run. The six-column sample has 1,945 unique IDs; all visible test keys are covered. Haiper values begin with two zeros, compatible with the pinned participant reference's default skip-two mAA configuration; deployed-scorer parity remains unverified. All 122 all-NaN poses correspond to outliers, and 1,823 non-outlier poses are finite.

## Recommendation

The time-budget blocker is resolved; no further rerun is required to complete this configured screen. Proceed to supervisor review of source grouping and evaluation protocol. Require the two dioscuri datasets together, treat remaining source relationships as provisional, and choose representative grouped development/confirmation partitions only after provenance review. Do not choose partitions using model scores. Preserve the same frozen baseline, split, evaluator, thresholds and conditions for all future improvements per AGENTS.md.

No split frozen and no baseline executed in this review. Remaining gates are supervisor-approved grouping/confirmation design, evaluator-version decision with explicit parity limitations, and a reproducible baseline environment.
