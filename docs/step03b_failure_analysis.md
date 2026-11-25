# Phase 3B failure analysis

No Phase 3B GPU execution has occurred. There are no measured full-development failure counts or domain comparisons yet.

The returned Phase 3A diagnostic merged both ETs scenes: all 19 true non-outlier images and one true outlier joined one model; two true outliers remained unregistered. This is historical evidence, not authorization to adjust matching or clustering.

After each completed inference, isolated evaluation derives true-scene/predicted-cluster contingency tables, merged clusters, split scenes, false outliers and missed outliers. Geometry diagnostics record verified graph component sizes, isolated images and empty graphs without using labels. Results also retain model counts, registration coverage, retained/verified pairs, per-stage costs and differences between primary and sensitivity scores. The merge table and chart enable dataset/domain comparisons without conditioning execution order on scores.

Failure classification:

- Missing files, resource errors and deadlines: infrastructure/interruption; incomplete, retry identical configuration and verified caches.
- CUDA OOM: RESOURCE_OOM, incomplete; return measured logs and stop instead of reducing settings.
- Successful stages with zero registered cameras: SCORED_ALGORITHMIC_FAILURE, complete all-outlier rows, both scores computed, retained in aggregates.
- Worker exceptions, invalid features, malformed poses or numerical problems: UNCLASSIFIED_REQUIRES_REVIEW until reproducibility and cause are established. Preserve logs/tracebacks; do not invent assignments or assume they represent a scored algorithmic outcome.

An interrupted stage has no valid checkpoint. Previous completed stages remain reusable. Unregistered individual images retain NaN poses and outlier assignments. An expensive incomplete SfM run is not a negative accuracy measurement. Return the partial report if its fixed cap is insufficient.

Potential Phase 4 hypotheses, not implemented: cross-scene edges may explain merging; weak verified connectivity may explain missing registrations; conservative outlier rejection may affect clustering and pose credit differently. Each would need paired dataset comparisons against the completed immutable baseline on the same benchmark, with runtime/failures/resources reported. Confirmation remains locked. No hypothesis may change this baseline's parameters.
