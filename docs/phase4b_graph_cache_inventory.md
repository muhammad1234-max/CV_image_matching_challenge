# Which ns64 data can the prototype read?

The four primary-seed archives matched every SHA-256 in reports/ns64_full/completed_archive_index.json. The inventory records the upstream revision, experiment/configuration hashes, archive hashes, every member hash and field completeness. Only two archives contain individual verified pairs. None supplies the calibrated geometry required for a weighted boundary test.

| Field | ETs, 22 | stairs, 51 | Haiper, 54 | Theater/Church, 76 |
|---|---|---|---|---|
| Complete opaque image IDs | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE |
| Verified pairs | PARTIALLY AVAILABLE, total 80 | PARTIALLY AVAILABLE, total 266 | AVAILABLE, 454 endpoint pairs | AVAILABLE, 616 endpoint pairs |
| Per-pair verified inlier counts | MISSING | MISSING | MISSING | MISSING |
| Database image/camera ID relation | MISSING | MISSING | MISSING | MISSING |
| Indexed keypoints, verified index pairs | MISSING | MISSING | MISSING | MISSING |
| Calibrated camera intrinsics and distortion | MISSING | MISSING | MISSING | MISSING |
| Valid camera-to-camera rotations | MISSING | MISSING | MISSING | MISSING |
| Mapper-ready database | MISSING | MISSING | MISSING | MISSING |
| Reconstruction information | PARTIALLY AVAILABLE | PARTIALLY AVAILABLE | PARTIALLY AVAILABLE | PARTIALLY AVAILABLE |

“Partially available” for reconstruction means final per-image predictions and registration summaries. Replayable camera records, observations and triangulated tracks are missing. Aggregate counts cannot determine which endpoints matched or the weight of an individual edge.

The administrative converter reads only the opaque image-name column from the already authenticated prediction file, validates the complete neutral-ID range, then reads verified_edges.json where present. It discards the other columns. Each derived topology cache has a separate trusted manifest in the inventory. Both imported successfully. ETs/stairs have genuinely unknown topology; materializing an empty graph would falsely assert zero verified edges. Outputs of the graph prototype cannot recover these missing measurements.

Workspace inspection covered loose geometry containers and top-level members of project ZIPs. Eleven organizer recovery archives contain keypoints, descriptors and matches for a different frontend. The residual SQLite and NumPy package-test files are unrelated. No eligible ns64 numerical parent was identified. Nested handoff ZIPs, final predictions and checkpoint receipts are not independent evidence of a valid pre-mapper cache. The scan records candidate file hashes for identification without importing them. Re-run from the project root with `python scripts/inventory_geometry_graph_v1.py`. [Machine inventory](../reports/phase4b/cache_inventory.json)
