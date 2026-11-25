# Execution audit

The approved archive, candidate configuration, protocol, and embedded notebook match their registered hashes. All 138 opaque image files matched their respective original byte digests in neutral order 20261111. The authorized dataset is imc2024_dioscuri_baalshamin in dev_a, source group heritage_dioscuri. Only this dataset's image package was opened. Identity restoration and scoring were never invoked. The worker text contains none of the 138 original filenames.

| Identity | Verified SHA-256 or revision |
|---|---|
| Phase 4I supervisor archive | caca1334b3e20bb5f3749263ec4192c15026d941139ccfd8b66d4b721d94e1de |
| Candidate configuration | 8134dd74471fe6bd25bf8ce3a891454f461e772d156715409f62977eb4c95b26 |
| Inference protocol | ec1af922c449ed0f68cc321cf972af772e491154ddaa36700a75632a24c4a690 |
| Neutral content permutation | 1b7bc97f6f54a2f5daa967a06e67211cd565f84d2d9403cd60f1faa50af543b0 |
| Opaque input ZIP | 2390cb5c85812982edfd47d73471e548fb70e5e8f4d4d76b35674951f089fd9c |
| Published ns64 revision | 07b5695b99a8ae4725d5d952f0065c2157fadd7d |
| Published pipeline YAML | e7829be60865192d3ddcc8555b94bc8678415e265a566d8ce8386ef17b0cc763 |

The CPU audit verified all Phase 4I archive members against their archived manifest and the workspace. It compared 1,810 published source files to their revision manifest and the clean runtime ZIP's inference source to the worker's embedded inventory. The historical weight receipt matches all seven primary hashes. Both DINOv2 sidecar pins agree across the source and runtime manifests. Verification of weight bytes on a future VM is still required. reports/phase5a/integrity.json records the remaining source, split, evaluator, environment, weight and organizer prediction pins.

## Notebook code review

| Path | Expected behavior and finding |
|---|---|
| Capture | The hook runs matching and verification, then intercepts the published mapper. SQLite backup checks integrity and logical equality. The wrapper checks image-file hashes at capture. The database image inventory check in parent_graph occurs later during partitioning. An early-return path with opaque.csv can seal a capture without a numerical parent. That path cannot establish the Phase 5A parent requirement. |
| Original | An audited unmasked copy is mapped with one thread, minimum model size 3 and maximum 25 models. The normal path validates prediction coverage, then exports indexed residuals with reciprocal track checks. An export error may leave a prediction and a valid_export=false receipt. The wrapper does not check for the presence of either receipt or prediction before sealing. |
| Trial | Eligible parents can trigger one cut. The mask removes all cross-side IDs from both matches and two_view_geometries in a transaction. The copy audit compares every retained row and all other tables and rejects extra pair records or unknown pair sources. No matching stage follows the deletion. A post-map audit compares the database with the masked input. The original branch resides in a separate directory. These are code-level findings; no 138-image mask was applied. |
| Decision | The selector compares corresponding observations and copies the selected CSV. The replay/scorer checks the selected bytes. An unavailable trial causes the original to be selected even when the failure came from an input-integrity check. |
| Resume and budget | Archive SHA-256, stage identity, earlier seals and payload hashes are checked. Completed stages debit recorded time. A valid parent can be reused without rematching. Hash verification accepts a self-consistent incomplete stage; a previously existing export ZIP also bypasses repackaging. An interrupted, unsealed attempt has no durable attempt/time debit in this wrapper. Manual custody of stage archives cannot establish mid-mapper resume. |

## Reproduced stop condition

The CPU probe called the frozen stage wrapper with a disposable predecessor, a local zero-exit child that produced no output, and a stand-in for Colab downloads and Linux monitoring. The original stage added time.json, sealed the folder, verified it and exported a ZIP. Its seal listed only time.json. Neither branch.json, prediction.csv nor errors.npz existed. A later restore can accept that same seal before discovering the missing scientific outputs. This probes incomplete-stage handling; it does not establish that the real mapper would have exited this way.

A second disposable fixture supplied a completed original and injected a trial input-digest error through the monitor interface. The frozen wrapper sealed an unavailable-trial receipt. Selection returned ABSTAIN and copied the original bytes exactly. That preserves the fallback prediction, but a protocol-integrity failure cannot support a scientific abstention on this dataset. Neither fixture is a prospective run.

**Required before launch.** A separately reviewed operational version must enforce a per-stage output schema before seal, restore and export. For the normal original require 138 unique valid predictions, the input audit, branch receipt and the matching observation archive when indexed export is declared successful. Distinguish explicit unavailable-export outcomes from missing files. Separate integrity failures from eligible resource/geometry fallbacks. Verify the exported archive against the stage and account for interrupted attempts before any retry. Exercise those paths with positive and negative cases. Preserve the frozen model and comparator identities and register revised runner provenance before use.

Reproduction from the project root uses `python -m scripts.phase5a_audit` on a copy of the audit output directory if preserving receipts. The existing targeted suite passed 20 tests; the full suite passed 322, including those 20. The additional CPU probe is separate from both counts. No frozen implementation was edited.