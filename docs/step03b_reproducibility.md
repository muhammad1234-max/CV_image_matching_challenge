# Phase 3B reproducibility

The full recovered executed source is retained under `snapshots/organizer_phase3a_verified/`. Every file matches the successful smoke manifest. Its snapshot fingerprint is `f57789a8bbd105de5a054040beb5289345ebb47e495b7e4ecf6b0fb25a792b3b`. Recovered executed `run.py` SHA256 is `f08d3da2e3b881489dfea6cf79a0b153b03e9a5dd0938a26c5635db69290d7aa`; current workspace's only historical difference was a later missing-input diagnostic. Exact old inference workers, function extraction, preflight and utilities are preserved, not edited.

The new orchestration is `organizer_smoke_verified_full_development_v1`. `configs/baselines/organizer_v1/full_implementation.json` is the authoritative complete SHA256 manifest and implementation hash. It pins executable source, original snapshot, model settings, environment/checkpoint locks, development inputs and isolated scoring metadata. The notebook embeds these exact bytes with an independent payload digest. No Git commit exists; content hashes are the source identity. The builder refuses to silently repin changed source.

Final prepared implementation SHA256: `d5d612c530d52089b6dfa7ab3589ccaee572d0ba6281b9bfb35ceba044e5ce93`. An earlier, unexecuted preparation manifest is retained in `snapshots/phase3b_preparation_history/`; it is not an experiment or comparator. No GPU result was produced under that draft.

Frozen benchmark remains `imc2025_local_v1`:

- Benchmark: `a78f0542413b8ac2ec8c262988ed5303c934490a2b31d2a45d3939dd6d3ab2a4`
- Split: `170546743202f21b9bcc98376ff09a6eae2da16ce833115478a976c4f46b2057`
- Evaluator: `c7ae94306ba0ae4ce9c00e9abc7e15bd99f3091fa063b5826b0dd567a6c3f261`
- Threshold manifest: `c6aec2fc2c8f9cb5f268f57333911f0835fcaba8aedc0eec67951ed729e68645`
- Inventory: `ad6c64f09b9590cd2ab1c67a7c472ce52aa761ab5c5297f30884faa9ad85ad37`

Input selection contains only dataset/image identities, randomized deterministically using seed 20261111. Images are copied to flat opaque names, converted to RGB and stripped of metadata except focal-length-in-35mm EXIF as in the smoke protocol. Inference receives these filenames and model settings; true labels/poses are read only by a separate process after predictions are finalized. All 1,563 selected identities match frozen development membership. No confirmation images or scoring metadata are selected.

Fidelity scope: exact successful smoke **inference** source/settings/weights, with generalized dataset/session orchestration and original-identity sorted CSV scoring. Historical smoke isolation already wrapped organizer functions, sanitized input names/metadata and serialized poses; this is not a claim of bitwise reproduction of every original organizer top-level notebook operation. Seed/CUDA settings do not guarantee identical GPU numerical results. Full package/driver records remain available for reviewing environmental drift outside the locked core. The Python 3.10 evaluator uses an equivalent streaming `hashlib.file_digest` shim; frozen evaluator bytes remain unchanged.

Stage keys bind implementation, configuration, environment, checkpoint content, original/sanitized image hashes, membership and predecessor artifact hashes. Resume verifies actual cached files. Completed results are exclusive and cannot be overwritten silently. Report inventory hashes prevent accidental corruption; they are not cryptographic attestations of remote execution. Image bytes and large cache files omitted from lightweight reports cannot be independently rehashed locally without those inputs. Local evaluation nevertheless independently verifies exact prediction coverage, poses, assignments and scores.

Only after all eleven datasets validate can `--register` create the exclusive `experiments/baseline/organizer_baseline_v1_dev_full.json`. It contains full results, sessions, both score variants and fold records validated against the existing experiment contract. Future improvements must preserve this comparator, source, environment, split and evaluator; corrections require an explicit new benchmark and reruns of all compared methods. Never edit a registered record.

## Batch 11 recovery and independent review

For batch 11, use `notebooks/step03b_batch11_recovery.ipynb` in the original pinned Kaggle environment. Attach both timestamp 1794728741629013008 archives as private inputs and execute all three cells in order. The launcher records its SHA256 and both input hashes in recovery_orchestration.json. It allows 14,400 seconds for reconstruction on this dataset. Use the same allowance for future comparisons. The prior 7,200-second timeout is an infrastructure interruption and its measured time contributes to compute accounting.

The final review reuses the first ten independent score checks after verifying the original report archive hashes, prediction hashes and dataset records. Batch 11 is scored from its predictions. Repeated Horn calls with identical numeric arguments reuse a copy of the first computed alignment.
