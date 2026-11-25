# Haiper reconstruction-feedback decision

SUCCESS — ACTIVE SPLIT IMPROVES LOCAL SCORE

The frozen participant and sensitivity evaluators agreed. Pass-through matched historical mAA 0.7444444444444445, clusterness 0.6352941176470588 and combined 0.685551871151113. Active mAA was 0.7388888888888888, clusterness 1.0 and combined 0.8498402555910544. The respective paired deltas were -0.005555555555555647, +0.3647058823529412 and +0.16428838443994132.

All 54 cameras registered in each branch. The 31-camera merged model became separate 15- and 16-camera models. All 23 control poses and camera centers matched exactly. The control retained 61,723 points, mean track length 4.0142 and mean reprojection error 1.2847 px. The affected parent had 73,841 points and 1.2220 px mean error. Its two successors had 37,757 and 26,072 points and 1.2400 and 1.1276 px errors. Separate reconstruction gauges prevent direct raw pose comparison across split models. There is no evidence here of improved pose accuracy or repeat-run stability.

Pass-through took 447.82844331399974 seconds and peaked at 767,288 KiB RSS. Active took 379.5216025609998 seconds and peaked at 734,516 KiB. Branch-specific GPU utilization was not recorded. Combined session elapsed time was 3039.4 seconds, below 5400. No matching or retraining was required. Independent replay of the SQLite edits reproduced the returned table fingerprints. Removing 240 crossing pair IDs from both pair tables eliminated 68 nonempty verified edges. All retained records passed the audit. The complete lightweight suite passed 273 tests.

This is one development dataset used to form the hypothesis. A prospective test should freeze a feedback rule on a different authorized development group before revealing scores. One testable rule accepts a proposed split only if registration is preserved, both sides reconstruct independently, and mean reprojection error does not increase on matched surviving observations. Measure track loss separately to avoid crediting a branch for dropping difficult points. Test whether this rule predicts a gain against the same pass-through baseline.
