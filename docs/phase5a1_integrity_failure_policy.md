# Failure handling

INTEGRITY_FATAL stops execution on mismatched content, provenance, required outputs or coverage. An input archive hash failure includes its expected and actual digest. The historical injected exception supplies only the message “fixture: trial input digest mismatch”. The wrapper preserves that exact diagnostic and stage without fabricating digest values.

EXECUTION_FATAL stops execution after unknown child exceptions, required-stage failures or ambiguous process exits. The Colab child entry point saves its exception type and reason before propagating the error. The monitor retains the worker log and resource samples in the attempt directory.

EXECUTION_RECOVERABLE is limited to verified no-eligible-geometry and a supervisor-side trial timeout after masked input preparation. Partial predictions or contradictory evidence invalidate the fallback. A child message that merely mentions timeout cannot authorize selection.

The journal records start before execution, elapsed seconds on caught failure, and the seal on completion. A lost VM with an outstanding start locks further execution with the entire reservation encumbered. Single-operator custody of the latest downloaded journal and its independent SHA256 is required across VMs. A lost journal requires review. A trial has one start. Restoring a registered receipt leaves the journal unchanged.