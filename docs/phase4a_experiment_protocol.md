# Paired evaluation protocol for Phase 4B

**Unit of comparison.** One original dataset, all of its images, and one frozen neutral input order. Benchmark imc2025_local_v1; seed 20261111. Use the pins in architecture_proposal_v1.yaml. The registered organizer baseline is organizer_baseline_v1_dev_full (11 datasets, 1,563 images, equal-dataset mean 0.4182297604006473). ns64 is an additional paired comparator on four datasets. A prepared 138-image input and a worker pin are insufficient to count a completed run. Archive identity and independently validated predictions determine eligibility.

### Conditions and fairness

1. Historical organizer and primary-order ns64. Preserve CSVs, receipts and all published numbers.
2. Instrumented, mask-disabled ns64. Capture, hash and replay the exact upstream inputs and mapper boundary. If the adapter changes runtime, edge values, mapping or output semantics, give it a new version and run it for every paired branch. A no-op should reproduce membership and pose conventions. On a shared captured database require identical edge inventory, inputs and mapper options. If poses differ across repeated unmodified mapping, quantify drift once on predefined ETs. Investigate any membership drift or mAA difference beyond the documented ETs order gap before attributing gains; do not use the better rerun. A changed comparator cannot silently replace historical ns64.
3. Active A, branch controls and ablations. Same bytes, same order, threads, hardware class, seed policy, limits and evaluator. Keep the first valid completion for each preregistered condition. Complete all-outlier predictions are legitimate outcomes. Keep interrupted/OOM runs and their costs as failures; missing datasets preclude the corresponding complete mean.
4. Precommit manifest with code revision/diff hash, proposal digest, environment, source, weights, order, image bytes, camera/match conventions, capture/export version, mapper options, evaluator and threshold hashes. Per-dataset receipt contains parent cache hash, timings, edge mask/abstentions and final predictions hash. Pair A with both the registered organizer and the matched ns64 control. Attribute a graph intervention only against the latter.

### Access and ordering

| Phase | Permitted observations | Decision |
|---|---|---|
| Debug | Synthetic camera graphs, known toy poses, edge-only archived development topology. Use identity permutations, inverse fixtures and negative cases. | Fix implementation errors only; freeze code, configuration, thresholds, gates, seed and run list before labels are joined. |
| Predefined first real dataset | imc2023_haiper, chosen because archived edges permit a direct mechanism check. Stage inputs with opaque IDs and neutral order in the isolated environment. | One instrumented baseline and one A branch; log geometry coverage, support and masks before scoring. If support is absent, record failure of mechanism, with no threshold trial. |
| Four-dataset paired comparison | ETs, stairs, imc2023_haiper, imc2023_theather_imc2024_church. Count the frozen Haiper run once. All influenced this research question. | Complete fixed list. Mark equal-weight four-dataset mean as descriptive subset only. A repair or redesign after seeing scores gets a new version and all four paired runs; the first analysis stays visible. |
| Broader development | All 11 predefined development datasets in their original dev_a/dev_b folds; compare on every common verified pair. | Any absent ns64 runs remain missing; report paired coverage, never extrapolate a full ns64 score. Keep heritage/dioscuri grouped. Full development requires all 11. |
| Protected confirmation | Two held-out source groups. Require a recorded supervisor release and frozen final experiment manifest, comparators and scoring environment before access. | One final evaluation, report separately. No tuning or model choice from confirmation or hidden test labels. |

Inference may access opaque images, neutral order, public checkpoints, legitimate pair observations and its own reconstruction state. The local administrator alone restores identities, joins true scene/outlier labels, and evaluates immutable predictions. Exclude original filenames, private identity maps, thresholds by scene, prior failure labels, administrative diagnostics and reference poses from the worker and any cache given to it. Compare intermediate artifacts by hashes. Distinguish provenance/duplicate checks from model features. Preserve both frozen evaluator variants. Do not use leaderboard feedback to choose cuts.

For B, cross-fit by existing source-group folds, holding out all four pilot groups from pilot fitting/calibration; see candidate B for the protocol. Keep byte-identical cross-dataset records in one group. Do not use transductive normalization across evaluated and fitted datasets. Existing pretrained visual models have incomplete training provenance; keep them fixed in each paired path and disclose possible overlap. A and C do not use scene labels for training, calibration or dividing features.

### Readout

Primary outcome is per-dataset change in frozen harmonic score, with equal-dataset mean on the declared set. Report separately mAA, clusterness, every paired delta, dev_a and dev_b means when complete, and group-weighted sensitivity. Record registration count over all images and registration recall over reference inliers. Report truth × predicted cluster contingency and counts of true scene pairs merged and true scenes spread across multiple predicted non-outlier clusters. Count outlier TP/FP/FN/TN, accuracy, precision and recall; zero denominator is undefined. A perfectly separated scene with an unregistered camera still has a coverage error. For ETs show secondary order 20261112 separately for both control and A, without best-order selection. For unchanged membership inspect aligned camera-center error by frozen thresholds and number of cameras crossing each threshold. Keep pose-error diagnostics restricted to the administrator.

Track CPU wall seconds for recovery, graph, mapping and evaluation; T4 worker/session wall and GPU allocated/reserved peak; process peak RSS, peak disk if instrumented, cache size and total failed-attempt cost. Distinguish shared upstream cost from incremental replay cost. Report per-dataset wall end to end as if deployed. On ten development source groups, summarize paired differences and leave-one-group-out influence. With four seen datasets and two confirmation groups avoid image-level significance claims.

### Predeclared ablations

| Variant | Change relative to otherwise identical inputs | What it tests |
|---|---|---|
| Pass-through | Preserve all verified edges | Drift and overhead of instrumentation |
| Topology-only | Always accept one eligible normalized cut | Whether graph partition alone accounts for gains; over-fragmentation |
| Local cycles | Remove a valid edge if at least four calibrated internal graph triangles are available and median rotation closure exceeds 15°; keep others | Local geometric reference independent of A’s cut-level vote |
| Independent edge pruning | Same proposed sides, internal checks and winning Q; remove only valid crossing edges outside the consensus radius; retain unknowns; apply only when A’s split gate fires | Is deleting the entire hypothesized boundary necessary? Report retained bridges. This is a simplified controlled comparison, not a reproduction of Gong or Xu. |
| All endpoints | Replace greedy disjoint support by highest-weight ≤32 crossing edges, allow repeated cameras | Effect of redundant evidence from one camera |
| Fixed uncertainty | Use a 15° radius with internal checks retained | Value of local angular tolerance |
| No internal check | Keep spanning-tree connectivity, set radius 15°, bypass the non-tree count and internal residual gates; compare against fixed-uncertainty variant | Harm from trusting poorly constrained frames |
| No minimum cross support | Retain endpoint balancing and angular tests; require only one valid crossing edge | Effect of abstention on weak boundaries |

Run CPU mask ablations first. Identical masks on identical parents can share a verified mapper output with an explicit receipt. Log all branches, including zero-effect branches. If feasible, add a published MA-SfM or external learned disambiguator comparison under a separate pinned adapter and report its costs; priority claims require a meaningful close-prior comparison.

### Decision criteria chosen before implementation

Advance from synthetic tests only if rigid synthetic scenes are retained under gauge changes, known incoherent test cuts are detected, sparse/degenerate cases abstain, and pass-through is verified. Advance from Haiper to the four only if an eligible boundary exists, produces a nontrivial active decision, and output is valid under the frozen schema. A no-op is reportable; it does not justify paying for three more full captures merely to lower a threshold. Advance to broader development only if equal-weight four-dataset combined delta versus the matched ns64 control is positive, at least one verified merge is corrected, neither ETs nor Theater/Church gains a false split, total inlier registrations fall by at most one across all four, and full-condition wall time is at most 1.25× matched control in aggregate. Also publish the deltas versus the organizer even if these gates fail. Reject the novel-component claim if controls explain the gain or it disappears on broader paired development. Confirmation requires a separate supervisor decision.
