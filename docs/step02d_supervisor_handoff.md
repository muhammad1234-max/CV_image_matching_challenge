# Phase 2D supervisor handoff

> Update after metadata receipt: **FROZEN**. All three original CSV hashes match the prior audit; full image membership and all hard constraints pass. The earlier draft/missing-input descriptions below are historical and superseded by `step02d_metadata_freeze_review.md`. No Phase 3 run is authorized by this freeze.


Phase 2D implementation is complete as a DRAFT; freeze remains blocked by missing authentic metadata. No baseline score exists. No model training, SfM, large download or Kaggle submission was performed. Phase 3 has not started.

## Deliverables

- `configs/benchmark/imc2025_local_v1/`: benchmark, source_groups, split_proposal, thresholds, evaluator, provenance and experiment_template JSON files.
- `scripts/benchmark_contract.py`, `evaluate_local_reference.py`, `prepare_benchmark_draft.py`, `validate_benchmark.py`, `register_experiment.py`, `export_benchmark_metadata.py`.
- `notebooks/02d_export_benchmark_metadata.ipynb` and `tests/test_benchmark_contract.py`.
- Six Phase 2D documents: validation_protocol, grouping_decisions, evaluator_specification, experiment_protocol, baseline_readiness and this supervisor_handoff.
- `docs/evidence/step02d_test_results.json` records actual lightweight suite results.

## Configuration and hashes

Version `imc2025_local_v1`; status DRAFT; seed 20261111. Primary: two grouped development folds and protected confirmation; intact datasets; all records retained. Hashes below fingerprint the current draft, not a frozen benchmark. Benchmark hash is canonical sorted JSON excluding its own benchmark_hash field. Proposal hash is canonical split proposal JSON. Split hash remains null because image membership has not been fabricated.

- benchmark_hash: `26e8ea072099eaddf14698db579d8a41d5654e42e91bc1e6f00b59a0457f9b5d`
- proposal_hash: `612595ad47ebb354acb37d277adfb228d10ee0c500ad53c28b5447bb53653064`
- dataset_inventory_hash: `ad6c64f09b9590cd2ab1c67a7c472ce52aa761ab5c5297f30884faa9ad85ad37`
- evaluator_hash: `c7ae94306ba0ae4ce9c00e9abc7e15bd99f3091fa063b5826b0dd567a6c3f261`
- threshold_hash: `c6aec2fc2c8f9cb5f268f57333911f0835fcaba8aedc0eec67951ed729e68645`
- split_hash: `None`

## Allocation

| Partition | Groups | Datasets | Records | Scenes | Outliers |
|---|---:|---:|---:|---:|---:|
| Development A | 5 | 6 | 728 | 13 | 85 |
| Development B | 5 | 5 | 835 | 13 | 3 |
| Protected confirmation | 2 | 2 | 382 | 4 | 34 |

Full 13-dataset mapping and provenance reasoning are in `step02d_grouping_decisions.md` and `source_groups.json`. Heritage and Dioscuri/Baalshamin share one source group and remain in A. No records are deduplicated. A trains on B; B trains on A; confirmation is excluded. Allocation balances development scene counts while preserving source boundaries, not outlier frequency. Small independent-group count and unknown pretrained overlap constrain claims.

## Verification and interpretation

38 returned artifact hashes and byte sizes match their manifests. Reported remote counts are 1945 records, 30 non-outlier scenes, 122 outliers and 1823 finite poses. Eight exact duplicate pairs support the mandatory merge; perceptual negatives do not prove independence. All 54 lightweight tests passed in 1.677 seconds (32 existing plus 22 new), with zero failures/errors, Python 3.13.7 and NumPy 2.1.2. Strict validator reports integrity PASS, status DRAFT, image membership NOT GENERATED. Tests cover grouping, leakage, coverage, outliers, determinism, overwrite protection, confirmation, thresholds, evaluator regressions, schemas, paired experiments, changed components and bounded metadata export. Synthetic tests are not benchmark scores.

Primary evaluator retains pinned participant overlap-first behavior; isolated sensitivity uses mAA-first/purity tie-break. All 30 threshold vectors preserve raw representations, including six zero entries excluded by skip-two. Hidden Kaggle parity and immutable upstream commit are unknown. Local source bytes and adapter are pinned. Immutable baseline registration, required hashes, exact sample identity/order checks and confirmation release records enforce reproducibility contracts; a frozen benchmark and Phase 3 approval are still required.

## Required next action

Run the supplied export notebook on Kaggle: accelerator None (CPU), Internet OFF, same ongoing competition input. Download `imc_benchmark_metadata.zip` into the project root. It exports only the original training-label, sample-submission and threshold CSV files with expected SHA256 checks and a 10 MiB cap. Then validate authentic per-image membership, review balance and freeze. Detailed commands and dependency gates are in `step02d_baseline_readiness.md`.

After freeze and supervisor approval, proposed first GPU run is an ETs development smoke test (22 records), followed by the complete two-fold baseline. Minimum categories are pinned Python/NumPy, CUDA-compatible PyTorch, selected feature/matcher code and weights, COLMAP/PyCOLMAP, plus exact environment and hardware records. Exact package versions and memory budget must be resolved against the selected implementation before execution. The smoke test cannot be registered as the scientific comparison baseline.

STOP at Phase 2D. Await metadata and subsequent supervisor approval before Phase 3.
