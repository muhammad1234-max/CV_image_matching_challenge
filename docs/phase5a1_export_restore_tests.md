# Export and restore evidence

Runner.export validates the live directory and all predecessors, creates a temporary ZIP, extracts it into a second temporary directory, validates the result, then publishes without replacing an existing file. An existing ZIP requires its separately retained hash and full validation. Invalid bytes are moved to a unique quarantine filename. A different valid same-name archive raises a conflict and retains its original path and bytes.

Runner.restore checks the external archive digest, the flat member inventory, expanded size, stage contract, registered seal and predecessor chain before moving the stage into place. A local valid checkpoint is reused only if its seal matches. Restore adds no time debit.

Tests in test_phase5a1.py cover round trips of all four stages, the time-only forged seal, unsafe and duplicate members, corrupted predecessor identity, incomplete archives and unchanged journal accounting. test_phase5a1_boundaries.py covers changed archive bytes, distinct valid colliding archives and malformed identities. The XML files identify individual test cases.