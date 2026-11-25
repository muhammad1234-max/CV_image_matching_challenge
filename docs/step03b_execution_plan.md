# Phase 3B execution plan

Status: PREPARED, awaiting human Kaggle execution. Zero of 11 development datasets completed in Phase 3B. The Phase 3A ETs result is not reused.

Scope: frozen dev_a (728 images, six datasets) and dev_b (835 images, five datasets). Confirmation (382 images, two datasets) and hidden test are excluded. Each original dataset is processed intact, never split by true scene. This pretrained pipeline does not fit on either training fold; fold names describe the frozen evaluation partitions.

Order is ascending image count, then dataset name, independent of scores. Batch IDs and counts are pinned in `configs/baselines/organizer_v1/full_development.json`.

| Batch | Stage | Dataset | Fold | Images | Possible pairs | Planning seconds, low–high |
|---|---:|---|---|---:|---:|---:|
| 01 | 1 | ETs | B | 22 | 231 | 232–926 |
| 02 | 1 | stairs | A | 51 | 1,275 | 1,025–4,101 |
| 03 | 1 | imc2023_haiper | A | 54 | 1,431 | 1,142–4,566 |
| 04 | 1 | imc2023_theather_imc2024_church | A | 76 | 2,850 | 2,191–8,766 |
| 05 | 2 | imc2024_dioscuri_baalshamin | A | 138 | 9,453 | 7,009–28,037 |
| 06 | 2 | fbk_vineyard | B | 163 | 13,203 | 9,728–38,914 |
| 07 | 3 | amy_gardens | A | 200 | 19,900 | 14,572–58,287 |
| 08 | 3 | pt_stpeters_stpauls | B | 200 | 19,900 | 14,572–58,287 |
| 09 | 3 | imc2023_heritage | A | 209 | 21,736 | 15,898–63,591 |
| 10 | 3 | pt_brandenburg_british_buckingham | B | 225 | 25,200 | 18,398–73,591 |
| 11 | 3 | pt_sacrecoeur_trevi_tajmahal | B | 225 | 25,200 | 18,398–73,591 |

These are unvalidated planning bands, not measured runtimes or confidence intervals. The low estimate is 30 + 21.33 n/22 + 9.28 n/22 + 29.13 P/226 + 9.75 P/231 + 131.39 (n/22)^2 seconds, P=n(n−1)/2. The high estimate is four times this expression. Pair selection may be sparse; SfM may be worse than quadratic. Smoke runtime alone cannot establish scalability.

Use one dataset per session initially. A fresh one-hour allocation with ten minutes reserved is sufficient under the planning band for batch 01 only. Six-hour sessions reserve 600 seconds, leaving 21,000 seconds: bands suggest batches 01–04 individually fit, but batches 05–11 may need stage resumption and may remain incomplete. Actual platform time/quota takes precedence over these limits. The runner records actual RAM, disk, GPU and stage timings before any extrapolation is accepted. Do not stack large batches based on an optimistic estimate.

Stage boundaries B–H are resumable with verified portable caches. SfM is capped at 7,200 seconds per attempt; interrupted SfM restarts from the verified geometry database, not an intermediate reconstruction. If a single stage cannot finish under its cap, return the partial report for supervisor review. Never shorten algorithm iterations or change resolution/keypoint settings to fit. No paid compute is provisioned.

Report ZIPs contain JSON/CSV/log evidence. Optional cache ZIPs contain embeddings, keypoints, descriptors, matches and a COLMAP database needed for stage resumption. They exclude images, weights and reconstruction binaries. Cache export stops above 3 GiB after saving the lightweight report; preserve the running session and report the storage bottleneck. Timestamp suffixes prevent silent archive replacement.
