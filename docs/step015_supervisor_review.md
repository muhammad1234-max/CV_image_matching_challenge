# Step 1.5 — Supervisor evidence review

Date: 2026-11-10. Reviewed all five Step 1 documents, diagnostic JSON/text, source manifests, and the saved metric implementation. Original evidence is retained unchanged.

| Finding | Correction or qualification |
|---|---|
| CLI absent from PATH | `python -m kaggle` works; version 2.2.4 and help captured. Installation is unnecessary. |
| Credential file present | Boolean inspection confirms legacy username/key fields. Credential loading is not server authentication. |
| Prior HTTP 401 | Retained as historical evidence. New CLI exits 1; its status parser found no HTTP code. Do not label the new result 401 without evidence. |
| Auth logging too aggressively suppressed | Secret-safe, but lost the precise new error. Report this limitation; do not repeat requests merely to improve logs. |
| Old audit script requests page size 1000 | Installed help says maximum 200. This is an independent script defect, not proven cause of the prior 401. Corrected future API listing to 200. |
| CLI submission template | Targets original competition; notebook/version placeholders are unfilled. Installed submit help uses positional competition and does not expose `-c`. Not executed. |
| Code/prose mismatch | Confirmed and explained by 12 tiny reference tests; no claim about hidden scorer behavior. |
| Scorer input validation | Participant copy filters GT to predicted datasets and uses dictionaries; completeness/duplicate validation must be independent. |
| Official source provenance | Participant copies are not authenticated current official scorer; organizer notebook is not assumed to contain it. |
| Source stability | Hashes pin retrieved files, not a full repository commit/environment. Full baseline revision remains unpinned. |
| Rules | Historical nine-hour/offline rules do not establish ongoing policy. Direct ongoing pages still unreadable. |
| Hardware | Usable RAM and available RAM were distinguished; old snapshot remains valid historically. Neither CUDA-tool absence nor CPU-only torch proves GPU absence. |
| Step 1 first-place recommendation | Revised to starter-first integration, then feasible top-10 reproduction, then expensive MASt3R work if justified. |
| Storage and GPU sizing | Proposed targets are estimates. No measured peak VRAM or dataset footprint supports a minimum-capacity claim. |
| Original verification report | Accurately says downloaded code was not executed in Step 1. Step 1.5 explicitly executes the reviewed, hash-checked reference on tiny fixtures. |

Completed only Step 1.5. No competition data transfer, model inference/training, heavy installs, Kaggle submission, or paid provisioning. See metric reconciliation for semantic findings and readiness for manual actions.
