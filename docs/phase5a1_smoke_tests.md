# CPU smoke evidence

The fixtures generate a small SQLite database containing 138 synthetic opaque names, two six-image graph components joined by a weak edge, synthetic residuals and 138 prediction rows. The geometry-unavailable variant removes the edges. All use the production Runner and Contract. The Colab adapter tests substitute only the native child execution and its environment, exercising the adapter's exception and timeout paths.

| Requested smoke | Case |
|---|---|
| A, G, I | test_roundtrip runs the full chain, exports each stage, reuses the ZIP, removes the synthetic live copy and restores it |
| B | test_colab_trial_adapter[timeout] produces an audited unavailable trial and the frozen ABSTAIN |
| C, E, F | test_time_only rejects child success, export, restore, and predecessor use even with an internally consistent seal |
| D | test_execution_failure and test_colab_trial_adapter[digest] stop before decision |
| H | test_false_receipt rejects a modified, resealed predecessor |
| J | test_stale quarantines an incomplete existing archive |

The targeted suite also checks masks, database inventory, duplicate or missing predictions, partial observation exports, required numerical artifacts, interrupted attempts, authorization, budgets, valid archive conflicts and scientific hashes. The duplicate ZIP member warning comes from deliberately constructing the invalid fixture.