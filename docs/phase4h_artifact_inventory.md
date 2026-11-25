# Haiper artifact inventory

| Field | Availability | Evidence |
|---|---|---|
| Original pre-mapper parent | AVAILABLE | Actual root-level SQLite bytes matched 9d088fd433baee7e4101ab25a8b7f5fedf405c8342c1bc779f88ef911c32051a. Schema and every logical table digest matched the frozen plan. |
| Opaque image IDs, original camera associations, keypoint coordinates, pairwise indices | AVAILABLE | Phase 4C parent, 54 images, 944,534 keypoints, 454 nonempty verified pairs. Pair counts were reread for this replay. |
| Original and pass-through pose/membership outputs | AVAILABLE | Phase 4C original export and Phase 4F registered return. Phase 4F recorded exact assignments, raw poses and frozen scores for both. |
| Active pose/membership output and mask audit | AVAILABLE | Authenticated active ZIP, prediction and input-table receipt. Its plan digest, parent logical records, complete options and forbidden IDs agree with the frozen plan. |
| Model membership lists | RECOVERABLE FROM SAVED ARTIFACTS | Join opaque CSV cluster indices to result.json model indices. Check membership cardinality against registered camera counts. |
| Model point counts, mean track length and reprojection error | AVAILABLE | Both returned result.json files. Aggregate errors have differing observation denominators. |
| Post-mapper optimized cameras, 3D coordinates, track-to-observation assignments and sparse models | MISSING | No branch sparse model in the returns or matching standard sparse-file names in workspace/archive inventory. Pre-mapper camera parameters and pairwise links cannot recover these. |
| Per-image surviving feature intersection, lost/gained triangulated observations, per-feature pixel residuals | MISSING | Requires branch-specific triangulation and optimized camera projections or captured indexed residual arrays. |

The archive inventory covered 104 ZIP directories including 27 nested archives and searched loose filenames. No standard sparse-model files were found. The Phase 4F branch exports intentionally included CSV/JSON and logs. A file-name search cannot authenticate an unlabelled reconstruction in an arbitrary payload.
