# CPU profile

The measured stages took 0.378 seconds for hashed import, 0.508 seconds for sparse graph construction, 0.280 seconds for connected-component track analysis, and 1.600 seconds for per-edge diagnostics. Process CPU time through these stages was 2.703 seconds. Peak process working set was 156.1 MiB. Timings exclude final serialization and administrator scoring. Processing uses bounded batches of 2,048 observed links; it does not construct a dense feature adjacency or enumerate all candidate triples.
