# Phase 4C supervisor decision

**BLOCKED — GEOMETRY COVERAGE**

The complete pre-mapper COLMAP snapshot for Haiper seed 20261111 was independently verified. All 54 images entered the graph. The database contains 944,534 keypoints, 1,431 raw pairs, 454 verified pairs and 607,136 verified correspondence indices. Source, configuration, checkpoint and permutation identifiers are recorded in the capture manifest. The numerical cache is the root-level database named by its SHA-256, 9d088fd433baee7e4101ab25a8b7f5fedf405c8342c1bc779f88ef911c32051a.

Local replay reproduced the graph decisions. All 454 pairs had uncorroborated focal priors and yielded zero usable rotations under the frozen rule. Candidate A and per-edge consensus abstained on both components. Triangle and pass-through retained both. Normalized cut proposed two splits removing 160 edges in the diagnostic mask.

The original mapper produced two models in 442.94 seconds. The run timed out during pass-through mapping after 1,393.72 seconds of a 1,453-second budget with a 60-second reserve. The last stage resource receipt recorded 6,806,888 KiB peak RSS and 5,367,882,752 bytes peak GPU allocation. The parity validator rejected incomplete two-branch execution. There is no paired score or active reconstruction result. Earlier local testing recorded 250 passes including 17 new tests.

For supervisor review, the lowest-cost parity follow-up is a mapper-only replay from the verified cache using the same images, options and environment. Even a parity pass would leave calibrated rotation coverage unresolved. Keep active testing on hold.
