# Kaggle execution instructions

Upload `notebooks/step03a_baseline_gpu_smoke.ipynb`. It embeds the audited source payload and frozen benchmark files; no separate project upload is needed.

1. Select a GPU (T4 if available); only cuda:0 is used. Set Internet OFF.
2. Attach `image-matching-challenge-2025-ongoing`, the original `oldufo/imc2024-packages-lightglue-rerun-kornia` dataset, and the three exact model versions listed in step03a_environment_and_differences.md. Utility source is embedded, so imc25-utils need not be attached separately.
3. Check mount paths in the settings cell. Legacy and current standard paths are supported. Change paths only to the corresponding authorized assets. Do not add confirmation/test data to the selected set.
4. Leave the 1200-second cap and source model settings unchanged. GPU_MEMORY_GIB defaults to 12 and can be reduced as an allocator cap; OOM then stops rather than changing the algorithm.
5. Run all cells. Stage A records the actual Python/GPU runtime. If it is not Linux Python 3.10, it stops with a compatibility report before attempting any package install or inference. This is expected on incompatible modern sessions. Do not force-install cp310 wheels in another Python version.
6. If a separately reviewed compatible Python 3.10 GPU environment is already available, set PYTHON to its interpreter. The harness does not create one or claim one is available on Kaggle. Original wheel hashes must match; no network installation is attempted.
7. Download `/kaggle/working/imc_baseline_smoke_report.zip`, including for a blocked/failed run, and place it in `D:\CV_Project`. The receipt specifies exactly which stages executed. If the runner cannot start, return the notebook error instead.

Stages B-H use actual model outputs. Stage I reports ETs-only primary/sensitivity diagnostic scores separately. Stage J means the receipt was saved, not that inference passed. A successful smoke run is not a registered baseline and does not authorize full folds.

Intermediate features/reconstructions stay under `/kaggle/working/organizer_smoke`. Re-running in the same session can reuse completed stages only when all input/config/source/checkpoint/environment and preceding-artifact hashes match. Changed checkpoint bytes fail and require a new run directory after review. Incomplete stages are moved aside, not reused. Repeated timed runs must be disclosed; do not use resume to evade the agreed experiment budget. Archive excludes images, model weights, feature arrays, reconstruction binaries and credentials; it includes JSON/log/CSV reports and original-identity predictions when they exist.

Stop after this one dataset. Do not submit to Kaggle, run full development folds, inspect confirmation scores or develop improvements. Supervisor review is required next.
