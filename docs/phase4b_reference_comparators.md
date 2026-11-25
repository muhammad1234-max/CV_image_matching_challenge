# Reference methods on a shared graph

Every method consumes the same immutable graph and emits the same pair-indexed mask format. Method IDs and configuration hashes are recorded. Pass-through is the default.

| Method | Decision | Removed edges on the 16-camera contradictory fixture |
|---|---|---:|
| pass_through_v1 | Keep every pair | 0 |
| normalized_cut_v1 | Split each eligible component on its connected weighted Fiedler sweep | 8 |
| triangle_v1 | For each measured edge, use every triangle with valid rotations; prune at ≥4 checks and median closure >15° | 0 |
| per_edge_consensus_v1 | Phase 4A ablation. When A’s split condition fires, prune individual measured crossing pairs beyond the winning radius; preserve unknowns | 7 |
| candidate_a_v1 | Remove the entire boundary when internal checks and disjoint consensus establish it | 8 |

The fixture contains 64 original edges. All eight contradictory bridges have disjoint endpoints, but individually participate in no triangle. A separate single-corrupt-edge clique test demonstrates that triangle pruning detects and removes the corrupted edge. On a rigid graph the topology-only splitting rule also removes eight genuine links; the full method retains them. A separate case holds exactly half of witnesses in consensus and retains the graph. Another case makes a crossing measurement unavailable and checks that individual-edge pruning preserves it. These are controlled geometric examples. No ablation has a measured effect on the competition metric. Per-edge pruning uses the shared gate specified in Phase 4A and is not an implementation of a published subgraph-pruning system. [Example decision masks](../reports/phase4b)
