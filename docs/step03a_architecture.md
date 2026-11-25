# Organizer architecture and implementation map

The runner loads only function definitions from the pinned organizer notebook. Its top-level loop, original ground-truth reader, submission writer and unpinned metric are never executed.

| Module/stage | Source behavior retained | Implementation |
|---|---|---|
| Input/A | Intact dataset image set | contracts.selection; run.prepare_images |
| DINOv2/B | Kaggle base/1, Transformers AutoImageProcessor/AutoModel; RGB32 [0,1] image; processor do_rescale=False; exclude CLS; max pool patch tokens; L2 normalize | original get_global_desc via reference.load |
| Retrieval/C | Euclidean cdist; <=0.3 candidate threshold; fallback first 20 sorted distances including self; skip self; skip final image as query; accept distance<1000; deduplicate sorted pairs; exhaustive when <=20 images | original get_image_pairs_shortlist, cached B embeddings injected |
| ALIKED/D | aliked-n16 default in acquired wheel; 4096 keypoints; detection threshold 0.01; resize 1024; float32 | original detect_aliked |
| LightGlue/E | Kornia LightGlueMatcher('aliked'); width/depth confidence -1; mp=True on CUDA; inherited filter_threshold=0.1; retain pairs with >=15 correspondences | original match_with_lightglue |
| Geometry/F | Original database importer, separate SIMPLE_PINHOLE camera per image; EXIF focal35 or 1.2*max(width,height); original pycolmap.match_exhaustive defaults | original import_into_colmap plus pinned utility files |
| Reconstruction/G | IncrementalPipelineOptions min_model_size=3, max_num_models=25; all other options from pycolmap 0.6.1 | worker.execute |
| Clusters/H | Each reconstructed model index becomes cluster ID; later model assignment overwrites earlier assignment for duplicated reconstructed image, as original; unregistered images become outliers | actual model outputs only |
| Pose output | cam_from_world rotation matrix and translation, row-major; nine decimals; unregistered R/t all NaN | contracts.serialize |
| Scoring/I | Frozen participant evaluator primary, mAA-first sensitivity separate; ETs-only post-inference diagnostic | score.py with unchanged Phase 2D adapter |
| Receipt/J | Config/source/checkpoint/input hashes; stage status, runtime, memory, counts, errors and diagnostics | run.archive |

DINOv2 base is checked for model_type=dinov2, hidden_size=768 and patch_size=14. Exact resize/crop/mean/std come from the mounted preprocessor_config.json; the source does not hard-code them. Actual JSON and its SHA256 are exported before inference. Do not claim independently inspected checkpoint bytes before the first mounted preflight. Images are processed one at a time; feature HDF5 files and correspondence pairs remain on disk. AMP is retained only where the original matcher enables it; ALIKED stays float32.

ETs contains 22 records, hence 231 possible image pairs and the original retrieval route is exercised (>20). ETs is dev_b, training for fold A and held-out evaluation for fold B. There is no dedicated training-only partition. Selection uses minimum intact dataset size, not scores. Its output score is an engineering smoke diagnostic, not a full-fold result or registered baseline.

Leakage controls: only ETs images are opened; protected confirmation and visible test images are not loaded. Ground-truth poses/scenes are confined to the ETs scoring CSV, never passed to inference workers. Original filename/scene hints are replaced by shuffled opaque PNG names before inference. Text/EXIF metadata are stripped except FocalLengthIn35mmFilm, the original camera prior. Images are losslessly re-encoded as RGB; this preprocessing/order change is explicitly an adaptation. A separate parent-side mapping restores original identities and CSV row order after inferred clustering. This is code-level separation, not an OS security sandbox against malicious code.

The frozen membership manifest includes all partition identities and is read for authorization/integrity only. No confirmation ground-truth CSV is embedded. Only the selected ETs labels and thresholds are included for post-run scoring; full train_labels.csv and sample_submission.csv are absent from the payload.

Failure behavior differs operationally: the original catches dataset errors and still emits NaN outputs. Our harness records the failing stage and stops; it never scores fabricated outputs after a failed reconstruction call. A successfully completed reconstruction with zero models legitimately produces all-outlier predictions. Partial/failed stages cannot be reused. A timeout kills the current child and preserves completed stage checkpoints; no automatic model-size/keypoint/resolution reduction occurs.
