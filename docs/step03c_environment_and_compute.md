# Environment and compute

ns64 specifies Python 3.11.11 for Kaggle, torch 2.5.1, pycolmap >=3.11.1, Kornia >=0.8 and NumPy <2. See upstream pyproject.toml for the full dependency graph. The organizer's Python 3.10.12, pycolmap 0.6.1 and Kornia 0.7.2 fail those requirements. ASMK and curope must be built for the target Linux ABI and CUDA environment. The release includes other adapted wheels in bundle/oss. Published mount aliases do not pin all weight bytes. Required active weights are listed in pilot.json, including the MASt3R retrieval head and codebook. DAD appears in general setup instructions but is absent from the selected matcher configuration.

Plan for a T4 compatibility probe first. Actual peak VRAM, host memory and throughput for ns64 are unknown. The author documents a two-GPU launcher. Matching scales with selected pairs, and CPU mapping depends on verified graph density. A defensible full-fold hour estimate requires stage timings on the target hardware and evidence from larger collections. The organizer's measured runtime is not an ns64 throughput estimate.


## Returned Kaggle diagnostic

The returned imc_top10_reproduction_pilot.zip matches the prepared plan and source archive SHA256. It ran for 15.95 seconds on Python 3.10.12 with two Tesla T4 GPUs (15,360 MiB each), torch 2.5.1+cu121 and Kornia 0.8.0. The source and ETs image checks passed. pycolmap, ASMK, curope and MASt3R distributions were absent. The combined probe stopped at the pycolmap import, so subsequent imports remain untested. Hardware PASS describes device discovery and does not establish full CUDA/model compatibility.

ALIKED and DINOv2 files were found with the expected prior-baseline hashes. SuperPoint, MASt3R, retrieval-head, codebook and ISC files were absent. No weights were loaded, no predictions generated, and no scores computed. The prepared launcher deliberately blocks at checkpoint inventory until original versions and environment are resolved. This receipt is diagnostic evidence. The end-to-end pilot remains pending. Review evidence is in reports/phase3c_pilot_diagnostic_review.json.
