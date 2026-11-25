# Step 1.5 — Execution environment and baseline sequence

Date: 2026-11-10. No installs, model runs or provisioning performed.

## Rule verification

| Topic | Evidence and status |
|---|---|
| Original notebook requirement | VERIFIED FROM OFFICIAL TEXT: “Submissions to this competition must be made through Notebooks.” |
| Original internet policy | VERIFIED FROM OFFICIAL TEXT: “Internet access disabled”. |
| Original duration | VERIFIED FROM OFFICIAL TEXT: CPU and GPU notebook caps are nine hours. |
| Original weights | VERIFIED FROM OFFICIAL TEXT: publicly/freely available external data and pretrained models allowed. |
| Original filename | VERIFIED FROM OFFICIAL TEXT: `submission.csv`. |
| Ongoing notebook requirement | Indexed page identifies a Community Code Competition; INFERRED notebook workflow, exact current requirement UNKNOWN. |
| Ongoing runtime, internet, external weights, filename | UNKNOWN; no readable current execution clause. Do not carry over original limits as fact. |
| Ongoing differences | VERIFIED FROM OFFICIAL TEXT via previously recovered index: “single leaderboard”; public/private splits combined, no prizes/points/medals/deadlines. Live UI state not verified. |
| Dataset/model input mounting | Public organizer notebook lists these inputs; mechanism exists. Ongoing permitted inputs and exact mount paths UNKNOWN. |
| GPU availability/quotas | UNKNOWN for this account; installed CLI offers `quota`, but no further authenticated calls were made after failure. |
| Rule acceptance | UNKNOWN; no acceptance action performed. |

Sources: [original official competition](https://www.kaggle.com/competitions/image-matching-challenge-2025), [ongoing evaluation](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/overview/evaluation), [ongoing rules](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/rules), [organizer notebook](https://www.kaggle.com/code/eduardtrulls/imc25-submission). Direct opens of both evaluation pages, ongoing rules, organizer notebook and notebook documentation returned no readable lines in this step. Indexed text is evidence of published wording, not a live authenticated policy snapshot.

## Revised reproduction sequence

| Candidate | Reproduction value | Dependencies/burden/risks |
|---|---|---|
| A. Organizer starter | First later integration check; does not satisfy top-10 requirement alone | DINOv2, ALIKED, LightGlue, pycolmap/COLMAP and utilities. Lowest expected engineering burden of these choices; exact notebook code/version still needed. [Source](https://www.kaggle.com/code/eduardtrulls/imc25-submission) |
| B. Full first place | Strong public-code target after simpler success | Multiple retrieval models plus MASt3R, detector keypoints, COLMAP, ASMK/curope builds. High estimated compute/dependency burden; code is public but no measured local runtime. [Report](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/REPORT.md), [dependencies](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/pyproject.toml) |
| C. Ninth-place refinement | More incremental feature-stack transition from starter | Author documents DINOv2 pair selection, multi-resolution ALIKED/LightGlue, rotation/crops, DBSCAN and RANSAC. Multi-pass/high-resolution processing remains expensive. Complete executable source/lock not identified; write-up inconsistently describes the direction of the 0.15 filter. Resolve similarity versus distance before reproducing. [Author write-up](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/ymg-aq-9th-place-solution-dinov2-optimized-filteri) |

Recommendation: A first, then inspect C's executable artifacts. C is simpler in model-family composition, not proven cheaper in runtime. If source cannot be recovered, use a pinned public-code top-10 candidate such as B; do not present a guessed reimplementation as exact reproduction. Any reduced B ablation must be labeled separately from its full final pipeline. No benchmark score is claimed.

## Compatibility and compute

Measured local hardware remains the Step 1 laptop: 2 GiB discrete VRAM, ~8 GB physical RAM, CPU-only torch. Python 3.13/Windows do not match the first-place configured Linux CPython 3.11 ASMK/curope wheels, torch 2.5.1, torchvision 0.20.1, NumPy <2 and transformers 4.47.0. Sources: [local evidence](evidence/environment.json), [upstream manifest](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/pyproject.toml). This is concrete incompatibility with bundled artifacts, not a blanket claim that all dependencies lack Windows/3.13 support.

Proposed future environment: isolated Linux x86-64/Python 3.11, pinned baseline-specific dependencies, supported NVIDIA driver and matching CUDA PyTorch wheels/toolkit for extensions. For B, CUDA12x-related dependencies in its manifest require compatibility review; no exact CUDA minor is established here. Start with an available 16–24 GiB GPU and 16–32 GiB RAM as engineering planning targets, not guaranteed minima. Prefer existing university or Kaggle resources. Inspect actual hardware/quota first; use a tiny authorized pilot before full runs. The laptop remains suitable for current tests and writing.

## Cache and temporary storage

Current TEMP/TMP resolve to C:. PIP_CACHE_DIR, HF_HOME, TORCH_HOME and UV_CACHE_DIR are unset (values never printed). Pip's effective cache drive was checked separately; see `step015_cache.json`. Full effective HF/Torch/uv cache locations were not queried. Before future installs/downloads set project-process variables to D: directories: `TEMP`/`TMP` -> `D:\CV_Project\tmp`; `PIP_CACHE_DIR` -> `D:\CV_Project\cache\pip`; `HF_HOME` -> `D:\CV_Project\cache\huggingface`; `TORCH_HOME` -> `D:\CV_Project\cache\torch`; `UV_CACHE_DIR` -> `D:\CV_Project\cache\uv`. Create these only when needed; no global environment setting or existing cache was moved. Use explicit D: download destinations. Reserve space for extraction and temporary builds, and remeasure immediately before use.
