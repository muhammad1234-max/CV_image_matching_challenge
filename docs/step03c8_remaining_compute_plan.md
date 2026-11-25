# Remaining development compute plan

Four measured T4 pilots at seed 20261111:

Images | Dataset | Total min | Matching min | Mapping min | Allocated GPU GiB | Worker RSS GiB
---:|---|---:|---:|---:|---:|---:
22 | ETs | 6.07 | 1.78 | 2.89 | 5.58 | 6.05
51 | stairs | 16.94 | 8.51 | 6.11 | 5.09 | 6.05
54 | haiper | 18.11 | 7.99 | 7.13 | 5.00 | 6.49
76 | theater/church | 36.07 | 11.40 | 21.62 | 5.60 | 6.04

The 76-image run spent 60% of its wall time in mapping. With 1.41 times as many images as haiper, mapping took 3.03 times as long. Use the closest measured workload (76 images) as the reference. The two scenarios scale each stage by image count and by possible pair count. These scenarios have no calibrated confidence bounds.

Dataset (one session each) | Source group | Fold | Images | Pairs | Source MB | Matching h | Mapping h | Total h | Scratch floor GiB
---|---|---|---:|---:|---:|---|---|---|---:
imc2024_dioscuri_baalshamin | heritage_dioscuri | dev_a | 138 | 9453 | 313.8 | 0.35 / 0.63 | 0.65 / 1.19 | 1.09 / 1.99 | 1.17
fbk_vineyard | fbk_vineyard | dev_b | 163 | 13203 | 98.4 | 0.41 / 0.88 | 0.77 / 1.67 | 1.29 / 2.79 | 1.00
amy_gardens | amy_gardens | dev_a | 200 | 19900 | 275.1 | 0.50 / 1.33 | 0.95 / 2.52 | 1.58 / 4.20 | 1.02
pt_stpeters_stpauls | pt_stpeters_stpauls | dev_b | 200 | 19900 | 153.2 | 0.50 / 1.33 | 0.95 / 2.52 | 1.58 / 4.20 | 1.00
imc2023_heritage | heritage_dioscuri | dev_a | 209 | 21736 | 3141.2 | 0.52 / 1.45 | 0.99 / 2.75 | 1.65 / 4.58 | 11.70
pt_brandenburg_british_buckingham | pt_brandenburg_british_buckingham | dev_b | 225 | 25200 | 166.5 | 0.56 / 1.68 | 1.07 / 3.19 | 1.78 / 5.32 | 1.00
pt_sacrecoeur_trevi_tajmahal | pt_sacrecoeur_trevi_tajmahal | dev_b | 225 | 25200 | 188.0 | 0.56 / 1.68 | 1.07 / 3.19 | 1.78 / 5.32 | 1.00

Seven remaining datasets total 10.76 / 28.39 GPU-hours under those scenarios, excluding setup, transfer, scoring and failed attempts.

Each proposed session keeps one original dataset intact, proceeds by count and preserves the listed partition. The heritage datasets share a source group. For each 138–225-image dataset, use a full T4 as the provisional GPU allocation. The four measured worker RSS peaks were at most 6.49 GiB and GPU reserved peaks at most 8.77 GiB. Future peaks are unknown; allow OS and model-loading headroom and measure free VRAM/RAM before each attempt. Scratch floors are max(1 GiB, 4 times source bytes) plus runtime, weights and source. Peak cache and peak disk were not measured. Heritage has the largest transfer and disk risk.

Every pair-count scenario exceeds 90 minutes. Obtain a session budget appropriate to the next dataset before launch. Preserve immutable stage receipts and transfer partial archives at the deadline. There is no verified numerical resume mid-SfM. Never merge an incomplete worker export. No remaining dataset has execution approval.
