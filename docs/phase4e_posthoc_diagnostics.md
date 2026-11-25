# Administrator-only Haiper diagnostics

The features and the two proposals were hash-sealed before joining the Haiper development labels in a separate process. Identity restoration was checked against the pinned historical export. No confirmation data was used. There were 386 within-scene and 68 cross-scene edges, with zero outlier-incident edges.

| Statistic | Within median [Q1,Q3] | Cross median [Q1,Q3] |
|---|---|---|
| Inlier count | 788.5 [259.5,2564.5] | 355 [119,721.75] |
| Neighborhood overlap | .579 [.478,.682] | .296 [.248,.346] |
| Cycle fraction | .097 [.053,.151] | .044 [.008,.063] |
| Two-witness fraction | .023 [.004,.052] | .005 [0,.009] |
| Clean multi-view fraction | .482 [.415,.559] | .653 [.605,.712] |
| Conflict fraction | .450 [.309,.547] | .235 [.153,.276] |

Flagging edges with zero exact witnesses yields 12 true positives, 20 false positives, 56 false negatives and 366 true negatives. Precision is 37.5% and recall is 17.6%. Neighborhood overlap already separates the central quartiles more clearly than cycle fraction. Across eight shared log2 count bins the within median cycle fraction exceeds the cross median in seven and ties in one. The cross-frequency-weighted within-minus-cross median difference is .042 versus .053 unadjusted. This controls coarsely for count, not for acquisition or image graph structure. Counts in the highest shared bin include only two cross edges. admin_aggregates.json preserves sample sizes and quartiles. Edges reuse cameras and features within one dataset, so an independent-edge confidence interval would overstate precision. Generalization and incremental predictive value beyond topology remain unestablished.

A separate nearest-log-count comparison without replacement matched all 68 cross edges. Within-scene support was higher in 41 of 68 pairs, with a median paired advantage of 2.26 percentage points. All 68 cross edges belonged to the 31-image proposed boundary. All 92 edges on the 23-image proposed boundary were within-scene. There is only one positive boundary, so edge-level separation cannot validate a boundary classifier.
