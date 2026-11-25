# Phase 3A source audit

Actual code was acquired through the public Kaggle kernel-pull API, not reconstructed from memory. Original API responses, notebook bytes and readable cell exports are preserved in `docs/evidence/phase3a_sources/`. Exact SHA256 values are in `configs/baselines/organizer_v1/source_audit.json`.

| Source | Acquired identity | Use |
|---|---|---|
| https://www.kaggle.com/code/eduardtrulls/imc25-submission | kernel 79815998, version 1 | Primary algorithm reference |
| https://www.kaggle.com/code/octaviograu/baseline-dinov2-aliked-lightglue | kernel 79905538, version 12 | Comparison only |
| https://www.kaggle.com/datasets/oldufo/imc2024-packages-lightglue-rerun-kornia | notebook datasetVersion 7884485 | Original wheels downloaded and hashed, not installed locally |
| https://www.kaggle.com/datasets/eduardtrulls/imc25-utils | notebook datasetVersion 11217117 | Original database.py and h5_to_db.py acquired and pinned |
| https://github.com/cvg/LightGlue | inspected HEAD 5a650c444718caffc2669dc0094cde908f42dd73 | Upstream context, NOT a replacement for the old wheel |
| https://github.com/colmap/colmap | inspected HEAD 137f21680c79d3f2fffbb1d11deb367162aa6f83 | Upstream context, NOT substituted for pycolmap 0.6.1 |

The notebook metadata records Python 3.10.12. Source kernel responses also record container image digests; those are preserved, not claimed to be selectable in a current Kaggle session. Dataset API downloads use current public endpoints, so wheel and utility content hashes are the definitive acquired identities; the relationship to historical dataset versions is reported from notebook metadata, not independently asserted byte-for-byte.

The expanded notebook changes ALIKED detection_threshold from organizer 0.01 to 0.05. Its helper default shortlist threshold differs, but both actual calls use 0.3. We retain the organizer's executable settings. Misleading original comments about EfficientNet are ignored: executable code loads DINOv2.

Licensing: LightGlue wheel metadata and upstream license say Apache-2.0; Kornia is Apache-2.0; pycolmap wheel metadata says BSD-3-Clause. Utility headers retain original copyright/license notices (h5_to_db Apache-2.0). Notebook license was not exposed in the acquired API metadata: public accessibility is not proof of a redistribution license. Keep this research bundle private and verify notebook/model terms before publication. Checkpoint terms are inherited from their Kaggle model pages and underlying projects; no checkpoint license is invented here. Attribution to Eduard Trulls, Octavi Grau and utility authors is retained.

No Git commit exists in this workspace and none is fabricated. `source_snapshot.json` fingerprints every executable file and configuration in the notebook payload, including unchanged frozen evaluator files. The notebook embeds a compressed, SHA256-checked copy of that payload. No weights or images are embedded. All baseline output remains an unexecuted architecture reproduction with documented input/environment adaptations until actual GPU evidence is returned.
