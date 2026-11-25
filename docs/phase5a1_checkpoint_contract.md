# Checkpoint contract phase5a1.checkpoint.1

Each stage contains contract.json, time.json, usage.json, heartbeat.jsonl and a seal.json covering its payload. The header binds the runner revision, experiment identity, published source revision, candidate configuration, permutation, stage, outcome and ordered predecessor digests. The seal retains the frozen evaluator format. The authoritative executable schema is src/runner_5a1/contract.py.

| Stage | Evidence checked |
|---|---|
| Capture | Numerical parent.db, backup and logical record hashes, SQLite integrity, foreign keys, 138 opaque names and keypoint counts, frozen mapper options, input order, worker identity, environment, assets and verify-matches receipt |
| Original | Source and input receipts, unmasked parent audit, valid 138-row pose CSV, proposal replay, model membership and indexed residual vectors aligned to keypoint counts |
| Trial | Same branch obligations with independently replayed masked-copy audit. An unavailable trial requires typed evidence and cannot include predictions or residuals |
| Decision | Valid predecessors, frozen selector replay, exact selection.json, selected bytes, pose validation and 138-row coverage |

Invalid depth or projection can produce OBSERVATIONS_UNAVAILABLE with a matching failure receipt. The next dependent stage stops. NO_ELIGIBLE_GEOMETRY requires replayed ineligibility. TRIAL_TIMEOUT requires an eligible proposal and a valid masked-input audit. COMPLETE requires the stage payload. Unknown files, symlinks and unrecognized failure reasons fail validation.