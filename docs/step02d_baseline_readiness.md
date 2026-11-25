# Baseline readiness

> Update after metadata receipt: **FROZEN**. All three original CSV hashes match the prior audit; full image membership and all hard constraints pass. The earlier draft/missing-input descriptions below are historical and superseded by `step02d_metadata_freeze_review.md`. No Phase 3 run is authorized by this freeze.


NOT READY for Phase 3. Phase 2D draft contracts are implemented, but the benchmark is not frozen and no baseline has been run. Blockers: original train labels, sample identities/order and threshold bytes; full per-image membership verification; freeze review; exact baseline code/config/weights/environment/evaluator pins.

Human action: upload `notebooks/02d_export_benchmark_metadata.ipynb` to Kaggle with the same ongoing competition data used in the completed audit. Set accelerator **None (CPU)** and **Internet OFF**. Run all cells. It reads only the three named CSV files, checks expected hashes, caps combined metadata at 10 MiB, and writes `/kaggle/working/imc_benchmark_metadata.zip`. Download that ZIP into `D:\CV_Project`. If a hash differs or multiple inputs match, retain the error and stop; do not bypass the check. The notebook does not read images, train, reconstruct or submit.

After reviewing/extracting the returned ZIP into a new evidence directory, run:

```powershell
python scripts/validate_benchmark.py
python scripts/validate_benchmark.py --metadata docs/evidence/benchmark_metadata_received --output configs/benchmark/imc2025_local_v1/image_membership_draft.json
python -m unittest discover -s tests -v
```

Membership materialization remains DRAFT pending review; it never auto-freezes. It checks authentic source hashes, exact dataset/scene/outlier counts, original duplicate endpoints and complete identities. Sample order is retained by preserving the original CSV. Freeze review then pins this artifact and all contracts in a new final manifest; no frozen files may be overwritten.

Proposed first GPU experiment, only after freeze and explicit Phase 3 approval: a bounded inference/matching/SfM smoke test on one intact development dataset (ETs, 22 records including three outliers), using a pinned participant-compatible baseline. This checks data wiring, failure accounting and evaluator integration; it is not the registered comparison baseline. Follow with both complete development folds, preserving all records. Confirmation stays excluded.

Minimum dependency categories: a pinned Linux Python environment compatible with the chosen baseline (Python 3.11 candidate), NumPy for evaluator, CUDA-compatible PyTorch, exact chosen feature/matcher package and weights, and matching COLMAP/PyCOLMAP builds. Exact versions and GPU memory requirement cannot be honestly finalized before selecting the baseline implementation and feature budgets. Record resolved lockfile, wheels/container digest, CUDA/driver, GPU model and weight checksums before execution. A single Kaggle GPU is a proposed smoke-test platform, not a tested resource guarantee. No installs, downloads, GPU runs or submissions were performed in this phase.
