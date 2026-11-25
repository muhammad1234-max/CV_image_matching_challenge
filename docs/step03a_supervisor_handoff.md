# Current status update

Actual Kaggle smoke execution has now passed. See [successful smoke review](step03a_successful_smoke_review.md) for verified results and limitations. The preparation-only status below is historical. Full-fold evaluation still requires supervisor authorization.

# Phase 3A supervisor handoff

Preparation is complete. Actual GPU/model execution has NOT occurred. The notebook is ready for runtime preflight; successful inference remains blocked on verification of a compatible Linux Python 3.10 GPU environment and mounted original checkpoints. No full folds, confirmation evaluation, submissions or novel models were run.

## Source and architecture

Actual organizer notebook version 1 (kernel 79815998) and expanded notebook version 12 (kernel 79905538) were downloaded through public Kaggle APIs. Original utilities and package wheels were acquired and hashed without local installation. Full URLs, notebook/container identities, SHA256 and licensing caveats are in `step03a_source_audit.md` and `configs/baselines/organizer_v1/source_audit.json`.

The primary implementation reuses organizer function bodies: DINOv2 base patch-token max pooling and L2 normalization; Euclidean shortlist threshold 0.3 with 20-neighbor fallback; ALIKED n16 with 4096 keypoints, resize 1024 and threshold 0.01; Kornia LightGlue with disabled adaptive pruning, CUDA mixed precision and >=15 retained matches; original pycolmap geometric verification and incremental mapping with min_model_size=3/max_num_models=25; model-index clusters and NaN/outlier failures. Expanded ALIKED threshold 0.05 is not adopted.

## Selection, isolation and differences

ETs is selected by smallest intact size: 22 records in dev_b, training for fold A and evaluation for fold B. No dedicated training-only dataset exists. It has 231 possible pairs. The choice was fixed before any model scores. Input names are randomized opaque PNG aliases; labels/poses never enter inference workers. Only post-inference ETs scoring sees ground truth. Original identities/order are restored for outputs; no confirmation/test images are loaded.

The harness is an architecture-faithful adapted reproduction, not demonstrated exact end-to-end reproduction. Declared adaptations include sanitized input metadata/re-encoding/order, bounded staged execution, detailed diagnostics/cache checks, fixed seeds, fail-fast behavior and the frozen local evaluator. Python 3.10 receives a SHA256-only file_digest compatibility shim outside the unchanged evaluator. Exact original torch/CUDA/Transformers versions and mounted checkpoint bytes are not yet verified.

## Dependencies and resource gates

Original wheel pins: kornia 0.7.2, kornia-rs 0.1.2, LightGlue 0.0 (exact wheel hash), pycolmap 0.6.1. Compiled wheels are cp310 Linux x86_64. The original notebook records Python 3.10.12. The harness requires a CUDA-enabled torch stack, torchvision, Transformers DINOv2 support, NumPy 1.x for the historical compiled stack, h5py, Pillow, tqdm, OpenCV and packaging. It records actual versions; no versions are invented where the source omitted pins.

Required mounts: DINOv2 metaresearch/base/1, ALIKED old ufo/aliked-n16/1, and LightGlue old ufo/aliked/1 (exact Kaggle identifiers are in the environment document). Checkpoint hashes are locked from mounted bytes before inference; no weights were downloaded locally. This does not independently authenticate the publisher hash. Package source bytes are verified against acquired SHA256. Missing or incompatible assets fail explicitly.

Default: one GPU, 12 GiB PyTorch allocator budget, 1200 seconds total target, unchanged model settings. Current Kaggle GPU compatibility is unobserved. Python 3.13 cannot load the original cp310 wheels; a modern incompatible session produces a BLOCKED report. T4 is a suggested smoke platform from the expanded source metadata, not a tested memory guarantee. No paid GPU provisioning or Windows GPU-package installation was performed.

## Deliverables

- `src/baselines/organizer/`: contracts.py, reference.py, preflight.py, worker.py, score.py, run.py.
- `configs/baselines/organizer_v1/`: smoke, assets, selection, source_audit, source_snapshot and receipt.schema JSON; ETs-only scoring CSVs.
- `notebooks/step03a_baseline_gpu_smoke.ipynb`: self-contained source payload, mounted original assets required.
- `scripts/build_phase3a_notebook.py`: deterministic payload builder.
- `tests/test_organizer_smoke.py`: 20 new synthetic/integrity tests.
- `docs/step03a_source_audit.md`, `step03a_architecture.md`, `step03a_environment_and_differences.md`, `step03a_kaggle_instructions.md`, `step03a_experiment_output.md`, this handoff, and test receipt in `docs/evidence/step03a_test_results.json`.

Frozen benchmark files were not modified. Benchmark hash remains `a78f0542413b8ac2ec8c262988ed5303c934490a2b31d2a45d3939dd6d3ab2a4`; split hash remains `170546743202f21b9bcc98376ff09a6eae2da16ce833115478a976c4f46b2057`.

Source snapshot hash: `b16bb8d4cc4302fb7368c908bd69864939838250a9451aabd0a127290213c635`.

Notebook SHA256: `7665f2fdd05729ed6535d0522f9c7738412809c65a6d4bfc5cfb291440c0fdee`.

## Verification and required human action

All 77 lightweight tests pass (57 existing plus 20 new), zero failures/errors, 2.736 seconds. Python 3.10 source syntax and notebook cell/payload structure pass. Tests verify frozen hashes, authorized dataset selection, opaque identity mapping, metadata stripping, serialization, missing reconstructions, complete outputs, cache invalidation, missing assets, failed subprocess/GPU-stage receipts and active timeouts. These are software tests, not model performance evidence. The frozen validator independently still reports PASS.

Upload the notebook, choose GPU T4 if available, Internet OFF, and attach ongoing competition data, original package bundle and exact model versions. Run all cells. If Python is incompatible, return the resulting preflight archive for environment review; do not force-install old wheels or silently upgrade algorithms. If a reviewed compatible interpreter already exists, the settings cell can point to it. Download `/kaggle/working/imc_baseline_smoke_report.zip` and place it in the project root even for a blocked run.

The archive is generated by actual notebook execution, not fabricated locally. It contains stage receipts, configuration/source/checkpoint hashes, logs, resources, reconstruction counts and permitted diagnostics when available; images, weights and credentials are excluded. Kaggle remote execution was not available in this preparation workflow.

Not ready for full development-fold evaluation: first obtain and review a successful real one-dataset run, resolve exact environment/checkpoint identity, then receive supervisor authorization. No baseline score or successful GPU result is claimed. STOP after Phase 3A preparation.
