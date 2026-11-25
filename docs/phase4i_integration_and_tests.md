# Integration and verification

The clean worker intercepts the original published incremental_mapping call after matching, two-view verification and pruning. It snapshots the database transaction and records the actual mapper-option summary and opaque image hashes. Replay opens an audited scratch copy and requires exact option equality before calling PyCOLMAP. It calls the published get_reconstructions and results_to_submission_df routines, so model assignment, world-to-camera pose convention and outlier formatting follow the source pipeline. The independent evaluator gets the original and chosen CSV bytes only after the output is sealed. Full return validation replays branch selection locally before the scoring process starts.

| Executed check | Result |
|---|---|
| Complete lightweight suite | 322 passed, including the 20 new targeted tests. See reports/phase4i/suite.xml and suite.log. |
| Targeted model and workflow tests | 20 passed. Coverage includes accept/reject/abstain, lost and gained rows, removal from the difficult quartile, registration changes, array corruption, no-op/insufficient evidence, bytes of the selected branch, foreign checkpoint/failure identities, safe archive restore and sample IDs with leading zeros. |
| Two-clique synthetic integration | A verified-count SQLite graph produced the expected one-cut trial. Two synthetic native-interface reconstructions used distinct local landmark IDs. Indexed export, decision and serialization selected the trial. Deleting one track backlink raised a validation error. The fixture tests mechanics only. |
| Haiper compatibility | Actual parent digest and logical tables matched Phase 4F. Recomputing the boundary reproduced both sides and all excluded IDs. Missing post-mapper track exports prevent a v1 score or a retrospective branch decision. |
| Native PyCOLMAP 3.11.1 CPU check | Loaded two genuine COLMAP sparse-format synthetic models with landmark IDs 7 and 900 and a twofold scale change. In each, the same two observed feature rows had zero pixel residual, and two untriangulated rows stayed missing. Reordering the database features was rejected. The native receipt is reports/phase4i/native_export.json. |
| Notebook inspection | Parsed every code cell and all generated worker stage programs for both targets. Verified every embedded inference input digest. Resume requires an externally saved download hash and a complete chain of earlier seals. |

A final integrity check matched the 4F return and plan digests, the 4H selected bytes and all freeze-file pins. The embedded kit contains no CSV, numerical parent, residual archive or checkpoint weights. Neither authorized worker text contains an original evaluation-image filename.

Local reproduction from D:/CV_Project using the project's lightweight Python environment:

    python -m scripts.phase4i_build
    python -m scripts.phase4i_audit
    python -m pytest tests/test_final_candidate_v1.py -q --basetemp=tmp/phase4i_user_target
    python -m pytest tests -q --basetemp=tmp/phase4i_user_suite

The builder derives the clean inference notebook from the existing hash-checked ns64 worker sources. Run it only when recreating the staged workspace, then compare the resulting freeze file to the delivered manifest. The archive is a review and reproduction overlay on the project. Historical numerical parents, competition images, checkpoint weights, private maps and ground truth are deliberately external to it. Repeating the audit changes its measured CPU duration.

The clean T4 notebook, new source-group reconstructions, paired administrative scoring on real returns, online execution time and Kaggle offline deployment have not been exercised in Phase 4I. The synthetic tests cannot establish pose accuracy or semantic scene separation.

To reproduce the independent native fixture, run python -m scripts.phase4i_native_probe from the project root in a Python environment with NumPy and PyCOLMAP 3.11.1. This check used Windows CPython 3.10 and NumPy 1.26.4. The frozen GPU inference environment remains Linux CPython 3.11.11.
