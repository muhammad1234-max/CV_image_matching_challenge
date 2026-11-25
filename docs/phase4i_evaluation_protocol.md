# Prospective development protocol

Protocol freeze is recorded in reports/phase4i/freeze.json and inference_pins.json. A clean notebook contains only the allowlisted inference kit and two opaque worker programs. The administrative machine retains mappings, thresholds and labels. Hyperparameters, zero training examples, seed 20261111, budget, selection rule, source manifests and frozen evaluator identities are fixed before new scores. The administrator checks both the downloaded SHA-256 values and the parent-linked seals, reconstructs the automatic decision, restores identity, and scores the sealed CSVs.

| Order | Dataset | Source group | Fold | Images | Rationale |
|---:|---|---|---|---:|---|
| 1 | imc2024_dioscuri_baalshamin | heritage_dioscuri | dev_a | 138 | Smallest remaining intact dataset in the worker index. An authenticated opaque archive exists. |
| 2 | fbk_vineyard | fbk_vineyard | dev_b | 163 | Next-smallest remaining group and a different development fold. Requires an opaque export. |

The eight known cross-dataset duplicates connect Heritage to Dioscuri. Heritage supplies no training, calibration, repeat or second independent group. ETs, Stairs, Haiper and Theater/Church informed design and appear only in historical results. The organizer has previously seen both planned groups during baseline evaluation. Independence here means prospective evaluation of the frozen intervention on additional source groups. There is no protected-confirmation claim.

**Paired comparison.** The frozen organizer prediction is checked by digest, rescored, and required to reproduce its registered scores. A new ns64 original is required on each group. Capture runs the verified original matching worker through verification and pruning. The staged original invokes its exact mapper options, unmasked parent and published postprocessor. Both methods use the same logical database, coordinates, image bytes, order and mapper. Original-pipeline drift or incomplete original blocks paired reporting. Archive provenance distinguishes this staged reference from historical registered workers. On a new group, direct monolithic-versus-staged numerical parity remains unmeasured.

**Ablations before scoring.** From exactly the same two maps, report the original/pass-through, the ungated graph proposal, the legacy v2 feedback gate, paired cost alone, and the full automatic gate. The cost-only arm tests improvement on both proposed sides without retention or control-image safeguards. The full arm includes those safeguards. If either map or indexed export is missing, mark the corresponding ablation unavailable. A fallback can still produce a valid automatic prediction. Keep any failed trial and its cost in the denominator.

**Administrative commands.** Place all four stage ZIPs from one dataset in one directory. Save their four printed download hashes as a JSON object keyed by capture, original, trial and decision. From the project root run the following once per dataset to a new output subdirectory.

    python -m scripts.phase4i_evaluate imc2024_dioscuri_baalshamin RETURNS PINS.json reports/phase4i_scored
    python -m scripts.phase4i_evaluate fbk_vineyard RETURNS2 PINS2.json reports/phase4i_scored

The runner emits both frozen metric variants, mAA, clusterness, combined, registration, merged clusters, split scenes, missed and false outliers, stage timing, memory, reason for fallback and paired differences against each comparator. Inspect comparisons with contingency tables privately. Share count summaries only. Use participant combined as the primary endpoint. Aggregate by equally weighted source groups, present all paired rows and planned/completed coverage, and mark partial means. Two groups permit a descriptive comparison only. A zero-delta fallback is an outcome. An incomplete original is a failed run with no paired score. Persist failure receipts beside the evaluation table. No leaderboard result enters model selection.


A capture or original failure can be registered without supplying predictions or calculating a score. Use the printed digest and preserve the downloaded ZIP. The pins-file position accepts the unused filename shown below in failure mode.

    python -m scripts.phase4i_evaluate DATASET FAILURE.zip unused.json reports/phase4i_scored --failure PRINTED_SHA256

The aggregate then lists the failed attempt, missing paired datasets, and the denominator of the available-group mean.
