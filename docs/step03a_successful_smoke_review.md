# Phase 3A successful GPU smoke review

The returned Kaggle report passes local review. All stages A-J completed on ETs alone. Both participant and sensitivity metrics recompute exactly from all 22 original-identity predictions, in the correct row order. Benchmark, split, evaluator, threshold and inventory hashes match the frozen benchmark. Ten included stage artifacts match their checkpoint hashes. Image bytes, large feature arrays and reconstruction binaries were excluded from the archive and could not be independently rehashed locally.

Runtime: Python 3.10.12, Tesla T4, PyTorch 2.5.1+cu121, CUDA build 12.1, NumPy 1.26.4, pycolmap 0.6.1, Kornia 0.7.2, kornia-rs 0.1.2, LightGlue 0.0, Transformers 4.47.0. Checkpoint content hashes and complete package versions are preserved in the received evidence. Actual maximum GPU allocated memory across workers was 1.58 GiB; maximum reserved was 2.69 GiB. These are PyTorch measurements, not total device consumption. Total wall time was 253.80 seconds, within 1200 seconds; SfM took 131.39 seconds.

| Measure | Result |
|---|---:|
| Images | 22 |
| Selected / possible pairs | 226 / 231 |
| Retained match pairs | 153 |
| Geometrically verified pairs | 138 |
| Reconstructed models | 1 |
| Registered images | 20 / 22 (90.91%) |
| Predicted outliers | 2 |
| Primary mAA | 0.2884615385 |
| Primary clusterness | 0.475 |
| Primary combined | 0.3589420655 |
| Sensitivity combined | 0.3589420655 |

All 19 non-outlier images registered, but both true scenes (9 and 10 images) merged into cluster0. One of three true outliers also joined cluster0; the other two were predicted outliers. This is an engineering success with a clear clustering limitation, not evidence of an improved model or a full-fold baseline result. No score-driven intervention has been made.

The returned source snapshot differs from current local code only in run.py: local missing-input diagnostics were added after this run's uploaded source. Removing only that known diagnostic block reproduces the returned run.py SHA256 exactly. The recovered executed source is saved as run_executed.py in the evidence directory. All other source snapshot entries match current files. The CPU environment check had previously stalled locally; this actual Kaggle run now establishes that the pinned Python 3.10 environment executes the entire pipeline.

Archive and machine-readable review: docs/evidence/baseline_smoke_received/local_review.json. Original root ZIP is preserved. No baseline was registered and no full development folds were executed. Phase 3A smoke testing is complete; next is supervisor review and authorization for a separately specified full-fold baseline protocol. Registration requires complete development results under pinned conditions. Protected confirmation remains excluded.
