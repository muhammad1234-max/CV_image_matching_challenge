# Synthetic acceptance cases

Expectations fixed before the first prototype execution. The two-block fixture uses strongly connected sides with noncommuting camera orientations and a weak matching of six or more distinct crossing endpoints. Contradictory alignments are separated by more than the configured tolerance. A cycle in a binary graph alone is not a measurement.

| Fixture | Expected outcome for Candidate A |
|---|---|
| One rigid scene | RETAIN; a common relative frame explains all witnesses |
| Two rigid groups with contradictory crossings | SPLIT and suppress every proposed crossing edge |
| Small noise on true rotations | RETAIN |
| Contradictory false bridge with high inlier counts | SPLIT if the weighted cut is still the planted boundary |
| Dense coherent repeated-structure alias | RETAIN even though the synthetic generating groups differ |
| Fewer than six distinct crossing pairs | ABSTAIN |
| Three internal non-tree checks | ABSTAIN |
| Two disconnected rigid components | Test each independently; zero edges added |
| Isolated camera | ABSTAIN for that component, keep node |
| Missing rotations | ABSTAIN |
| Nonfinite or improper rotations | Reject input before decision |
| Reversed pair order with transposed rotations | Identical canonical decisions and masks |
| Change of gauge | Same residuals and decision under local frame gauge changes |
| Renumbering/permutation | Identical mapped-back decision where the geometric answer is unique |
| Equal-weight symmetric graph | ABSTAIN for repeated Fiedler eigenvalue; stable under permutation |
| Near-degenerate eigenspace | ABSTAIN at the pinned gap |
| Many pairs sharing an endpoint | ABSTAIN when fewer than six disjoint witnesses survive |
| Sparse true bridge | ABSTAIN, retain bridge |
| Exact half consensus | RETAIN |
| Unsupported internal frame or high internal closure errors | ABSTAIN |

Topology-only input keeps the verified endpoints with unknown counts. The active and normalized-cut branches must abstain instead of fabricating unit weights. A no-op keeps the exact canonical pair set on every accepted input.


### Observed validation

The preregistered outcomes above held. The complete lightweight suite passed 233 cases, including 70 new cases. The first targeted run exposed a SciPy test-input shape error in the zero/180-degree fixture. Supplying the one-axis batched angles in the expected column shape fixed the setup; the expected geometric outcomes and thresholds were unchanged. The final suite also covers exact half support, stable equal-cost sweep selection, multiple eligible components, sparse solving above 256 nodes, mask replay and topology-only archive import. Full per-case results are in reports/phase4b/test_results.xml. The public sensor convention still requires an independently validated capture or recovery adapter on real data.
