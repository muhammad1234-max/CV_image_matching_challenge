# ETs resource use

| Measurement | 20261111 | 20261112 |
|---|---:|---:|
| Wall time, seconds | 364.174 | 355.410 |
| Retrieval, seconds | 31.78 | 31.06 |
| Point-tracking matching stage, seconds | 106.99 | 106.54 |
| Geometric verification, seconds | 10.86 | 10.18 |
| Incremental mapping, seconds | 173.53 | 167.11 |
| Shortlisted / retained / verified pairs | 231 / 80 / 80 | 231 / 80 / 80 |
| Peak GPU allocated, GiB | 5.58 | 5.58 |
| Peak GPU reserved, GiB | 8.77 | 6.96 |
| Peak worker RSS, GiB | 6.05 | 6.05 |

All timed stages passed. Raw timings include the near-zero wrappers, which do not measure separate feature branches inside the active matcher. The historical total includes local scoring. Neutral wall times include preflight and exclude separate administrative scoring. Stage timings and these wall times have different coverage. RSS is the worker high-water mark. Both runs completed within 1800 seconds.
