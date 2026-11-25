# CPU profile on synthetic geometry

Each size ran in its own Python process on the project laptop, with BLAS/OpenMP threads requested at one. The input contains deterministic noncommuting rigid camera rotations, two well-connected subgraphs, and crossing witnesses. Three repetitions per size use the same serialized payload. Values are median milliseconds. Total includes graph decoding/validation, component traversal, decision logic and mask hashing. It excludes fixture generation, module startup, final receipt file I/O, and any image processing. Stage medians need not add to the total.

| Cameras | Verified edges | Import | Spectral proposal | Internal frames | Boundary vote | Total | Process peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 22 | 121 | 7.77 | 1.94 | 2.55 | 0.54 | 14.11 | 62.5 |
| 51 | 326 | 16.95 | 2.98 | 7.89 | 1.10 | 30.89 | 64.0 |
| 76 | 514 | 27.90 | 4.14 | 11.25 | 2.02 | 47.93 | 64.5 |
| 138 | 979 | 58.27 | 5.89 | 21.76 | 3.10 | 91.15 | 66.4 |
| 225 | 1,631 | 89.47 | 10.31 | 38.62 | 4.88 | 152.67 | 70.3 |

The Windows process peak working set includes Python, SciPy, fixture generation and retained receipts. It is not incremental model memory. The 225-node importer consumed about 59% of median total time, including JSON and per-edge SO(3) validation. Frame propagation and recording each loop residual were the next largest costs. The graph uses sparse adjacency. For ≤256 nodes a bounded dense Laplacian eigensolve requests only three eigenpairs; at 225 nodes one dense float64 matrix occupies about 0.39 MiB. Above this limit the code uses deterministic-start sparse eigsh. The 270-node regression exercises that branch. Endpoint comparison uses at most a 32×32 distance matrix. Triangle control can cost more on dense cliques and was excluded from this Candidate A profile. These measurements say nothing about matching, calibration recovery, BA or COLMAP runtime.

Reproduce with `python scripts/profile_geometry_graph_v1.py`. NumPy 2.1.2 and SciPy 1.17.0 were used here. The report records Python/OS/CPU identity, individual repeats, actual serialized bytes and process memory before and after. [Raw measurements](../reports/phase4b/cpu_profile.json)
