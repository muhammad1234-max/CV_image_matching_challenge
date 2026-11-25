# Kaggle notebook readiness

The six-column writer is implemented and tested on a synthetic sample. The runnable notebook in this package targets a clean development Colab VM. Production Kaggle compatibility, offline resource limits, hidden input layout and competition acceptance remain unverified. A scored submission requires a separate deployment and approval.

| Expected notebook cell | Input, operation and failure behavior |
|---|---|
| 1. Offline environment | Mount a private wheelhouse for exact CPython 3.11.11, PyTorch 2.5.1+cu121, PyCOLMAP 3.11.1, the compiled ASMK/curope extensions and the full pinned package freeze. Install from mounted paths with pip --no-index. Hash source, wheels and environment. The Colab bootstrap downloads packages and is unsuitable for this offline cell as written. |
| 2. Assets | Resolve the local aliases for SuperPoint, ALIKED, MASt3R metric, its retrieval weights and codebook, ISC and DINOv2. Verify the frozen manifests, including DINOv2 config and processor assets. Set HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE. Abort on a missing asset. |
| 3. Input discovery | Resolve exactly one mounted competition root by the actual sample_submission.csv and test tree. Parse image_id as text and keep (dataset,image) and row order. Handle repeated basenames in different datasets in separate jobs. Check duplicates, unknown files and missing rows. |
| 4. Budget | Measure actual GPU, wall time, scratch and cgroup headroom. Reserve time to write and validate the entire CSV. Schedule one original per dataset first and at most one split trial under a documented production budget. The development model cap is 225 images. Larger groups would require a separately validated resource policy. |
| 5. Inference and clustering | Create a label-free opaque view of each test dataset and retain the inverse name association only for output. Run published matching, capture, unmasked mapping and the frozen candidate. Each registered camera inherits its selected COLMAP cluster and optimized world-to-camera pose. An unsupported or incomplete trial returns the completed original. Missing original predictions block export. |
| 6. Serialization | Restore display filenames and dataset values, use real sample image_id strings, flatten R row-major and t into semicolon-separated vectors, preserve the outlier sentinel, and match every sample row once. |
| 7. Validation and output | Write /kaggle/working/submission.csv atomically. Require exact ordered header, bijection on sample IDs and image keys, valid finite rotations and translations for registered views, all-NaN 9/3 for outliers, and enough storage. Record file hash, row count, resources and fallbacks. |

Header verified from the archived competition contract:

    image_id,dataset,scene,image,rotation_matrix,translation_vector

After collecting selected rows with the real dataset and image names, the implemented final cell is:

    from pathlib import Path
    from src.novel_model.final_candidate_v1.submission import write_submission
    write_submission(sample_path, selected_rows, Path("/kaggle/working/submission.csv"))

Before a production launch, build the offline environment and run a public/synthetic dry run with network disabled. Confirm the active competition rules and submission availability separately. No Kaggle submission was attempted.
