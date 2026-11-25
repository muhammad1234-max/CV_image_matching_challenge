# Smoke receipt contract

The smoke receipt is deliberately separate from the immutable full-baseline experiment registry. `kind=smoke_test`, `not_registered_baseline=true`. It may be PASS, BLOCKED or FAILED. Every stage A-J has a status; unexecuted stages remain NOT_RUN. Receipt creation J can pass after an inference failure.

Required receipt fields: frozen benchmark/split hashes, dataset=ETs, role, full configuration and hash, total runtime and stages. Once preflight verifies them, source snapshot, evaluator, threshold and inventory hashes are recorded. Stage A runtime records Python, packages, torch, CUDA, GPU/VRAM, driver output and actual checkpoint hashes. Missing setup information remains unavailable, never fabricated.

Each inference result includes elapsed time, GPU peak allocated/reserved memory and CPU peak RSS where measurable. Retrieval records embedding shape and pair counts; extraction records per-image feature counts; matching records raw and retained per-pair counts; geometry records verified pairs/inliers; SfM records model sizes and unique registrations; H records outliers inferred from reconstruction. Registration coverage is registered unique images / 22 including all original outliers in the denominator.

I/metrics.json labels metrics as local ETs engineering diagnostics and separates participant and prose_sensitivity variants. All 22 expected identities must appear exactly once; missing reconstructions serialize as outliers and fully NaN poses only after a successful reconstruction stage. Original training-row order is restored. No submission.csv is created and no submission API is called.

Cache checkpoints bind stage artifacts to input pixel/source hashes, numerical config, executable source, checkpoint content, environment and preceding-stage artifact hashes. SHA mismatches invalidate reuse. Errors preserve tracebacks and available memory information. Timeout output cannot masquerade as a completed stage. A generated archive from synthetic unit tests is temporary test data, not a reported model execution.

Before the first full baseline can be registered, both complete development folds must actually run under supervisor authorization, the exact GPU environment/weights/code must be locked, per-dataset metrics and failures validated, and the immutable baseline reference registered using the Phase 2D contract. Phase 3A does not satisfy those requirements.
