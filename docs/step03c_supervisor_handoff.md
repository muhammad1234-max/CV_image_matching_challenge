# Supervisor handoff

**Preparation status: preflight only.** Verified source and baseline identities, selected ns64 and packaged the bounded diagnostic notebook. No Kaggle inference or full-fold launch has occurred.

Import notebooks/step03c_top10_reproduction_pilot.ipynb into a fresh Kaggle notebook, enable a T4, keep Internet off, attach the competition and attach configs/baselines/top10_reproduction_v1/ns64_pilot_source.zip.bin as a private file with its exact name. Attach any available original model bundles. Run its single cell and download /kaggle/working/imc_top10_reproduction_pilot.zip. A BLOCKED receipt is expected. The archive inventories available weights, hardware and imports up to the first failure. No predictions are available at this stage.

The next decision is whether to provision ns64's original Linux wheel and checkpoint inputs for the same ETs pilot. Before inference, verify exact versions and load the models, integrate the published entry point with label-free dataset staging and the frozen scorer, and test output ordering, poses and paired scoring. Separate adaptations require explicit documentation. Files comprise the two upstream snapshots, the manifests and archive in configs/baselines/top10_reproduction_v1, the isolated preflight module, the pilot notebook, tests and six Phase 3C documents.


## Returned Kaggle diagnostic

The returned imc_top10_reproduction_pilot.zip matches the prepared plan and source archive SHA256. It ran for 15.95 seconds on Python 3.10.12 with two Tesla T4 GPUs (15,360 MiB each), torch 2.5.1+cu121 and Kornia 0.8.0. The source and ETs image checks passed. pycolmap, ASMK, curope and MASt3R distributions were absent. The combined probe stopped at the pycolmap import, so subsequent imports remain untested. Hardware PASS describes device discovery and does not establish full CUDA/model compatibility.

ALIKED and DINOv2 files were found with the expected prior-baseline hashes. SuperPoint, MASt3R, retrieval-head, codebook and ISC files were absent. No weights were loaded, no predictions generated, and no scores computed. The prepared launcher deliberately blocks at checkpoint inventory until original versions and environment are resolved. This receipt is diagnostic evidence. The end-to-end pilot remains pending. Review evidence is in reports/phase3c_pilot_diagnostic_review.json.
