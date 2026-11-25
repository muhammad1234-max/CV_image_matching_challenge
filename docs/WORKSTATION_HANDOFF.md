# Workstation handoff

## Current research state

Phase 5A.6 Kaggle preflight passed and its historical Dioscuri/Baalshamin CAPTURE completed in 1,648.895 seconds. Its checkpoint was lost before ORIGINAL. Phase 5A.7 private fixture restore and GPU preflight passed. The recorded recovery balance is 12,751.105 seconds and the authorized recovery remained unused at the last verified receipt. A fresh Prompt 36 rerun was discussed. No completion receipts for that rerun were verified during this handoff.

- [Production runbook](phase5a6_supervisor_handoff.md)
- [Recovery and persistence workflow](phase5a7_persistence_runbook.md)
- [Frozen independent evaluation](phase4i_evaluation_protocol.md)
- [Recovery budget ledger](../reports/phase5a7/ledger.json)
- [GPU preflight release](../reports/phase5a7/gpu_preflight_release.json)

## Set up the second PC

Clone this repository into a new folder and use an isolated Python environment. Use the pinned environment and wheel manifests in the phase reports for runtime setup. Normal GPU execution is on Kaggle. Run the CPU suite with `python -m pytest tests -q` after installing its dependencies and restoring the referenced private fixtures. Preserve original archive and journal filenames and verify them by SHA-256.

Transfer separately through authorized private storage from the old workstation or the existing private Kaggle inputs:

- `kaggle_upload/phase5a5_private_inputs/` contains ns64_runtime.bin and dioscuri_opaque.bin.
- `kaggle_upload/phase5a6_initial_custody/` contains the initial journal for a distinct Prompt 36 run.
- Scoring labels, identity mappings, organizer baseline predictions, and original experiment archives must be restored to their original relative paths from the trusted workstation. Keep them out of inference mounts.
- Restore historical ZIPs and numerical checkpoints only for the specific run or audit that needs them. The recovery fixture is synthetic and has no place in the numerical-parent chain.

Use the latest authenticated journal and corresponding checkpoints for each run. Keep the original project available until the private artifacts and CPU workflow have been verified on the second PC.
