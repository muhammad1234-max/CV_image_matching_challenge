# Source grouping and provenance

> Update after metadata receipt: **FROZEN**. All three original CSV hashes match the prior audit; full image membership and all hard constraints pass. The earlier draft/missing-input descriptions below are historical and superseded by `step02d_metadata_freeze_review.md`. No Phase 3 run is authorized by this freeze.


All 38 files listed by the three returned archive manifests passed local SHA256 and byte-size verification. This establishes consistency with returned manifests, not independent verification of Kaggle data. The complete 900-second-budget overlap rerun supersedes the incomplete 120-second run. It reports 1945 images screened, 1713342 cross-dataset comparisons, eight exact-byte pairs, no additional dHash candidates at distance <=5 and no image errors. dHash negatives do not establish independence. No original image bytes were rehashed locally.

| Dataset | Candidate group | Proposed evaluation | Images | Outliers |
|---|---|---|---:|---:|
| ETs | ETs | B | 22 | 3 |
| amy_gardens | amy_gardens | A | 200 | 0 |
| fbk_vineyard | fbk_vineyard | B | 163 | 0 |
| imc2023_haiper | imc2023_haiper | A | 54 | 0 |
| imc2023_heritage | heritage_dioscuri | A | 209 | 61 |
| imc2023_theather_imc2024_church | same as dataset | A | 76 | 0 |
| imc2024_dioscuri_baalshamin | heritage_dioscuri | A | 138 | 24 |
| imc2024_lizard_pond | same as dataset | Confirmation | 214 | 34 |
| pt_brandenburg_british_buckingham | same as dataset | B | 225 | 0 |
| pt_piazzasanmarco_grandplace | same as dataset | Confirmation | 168 | 0 |
| pt_sacrecoeur_trevi_tajmahal | same as dataset | B | 225 | 0 |
| pt_stpeters_stpauls | same as dataset | B | 200 | 0 |
| stairs | stairs | A | 51 | 0 |

Mandatory merge evidence: `docs/evidence/contract_audit_rerun_received/dataset_overlap_candidates.csv`, eight rows linking Dioscuri archive 0003/0004/0010/0015/0026/0230 and img 0055/0105. Preserve duplicate records in their original datasets: grouping prevents boundary leakage, it does not deduplicate competition data. Other components are unchanged because additional relationships have not been established. See the returned `source_provenance_review.md` for source-family evidence and limitations. Broad collection names alone do not justify a claim of independent domains.

Locally confirmed: artifact hashes, dataset/group arithmetic, threshold representations, pinned evaluator behavior in synthetic tests. Reported remote evidence: labels, image counts, pose integrity, byte duplicates and screening completeness. Assumptions: two confirmation groups provide a useful but narrow final check; source components approximate independence; pretrained overlap remains unknown. Missing: original labels, sample identities/order and original threshold CSV bytes. The image inventory lists paths and sizes but cannot replace the scene-labelled record table. `provenance.json` pins the local reports; no fabricated image membership has been generated.
