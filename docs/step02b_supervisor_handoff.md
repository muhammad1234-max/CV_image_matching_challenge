# Kaggle-native audit handoff

Notebook construction and local QA completed; actual Kaggle execution is user-operated and has not occurred. No local authentication was retried, credential configuration changed, model run, dependency installation, GPU provisioning, large download or submission performed. The provided submit template was not executed.

New files: notebooks/step02b_kaggle_dataset_audit.ipynb, scripts/kaggle_dataset_audit.py, tests/test_kaggle_audit.py, the three step02b documents, and QA evidence. Earlier project implementation/docs/tests are preserved; before/after hashes are recorded in evidence/step02b_integrity.json.

Architecture: self-contained offline NumPy/standard-library code; input discovery and overrides; one filesystem inventory plus bounded metadata parsing; disk-backed composite-key checks; aggregate CSV/JSON/SVG; source-hashed thresholds; provisional validation review; allowlisted size-bounded ZIP. Missing or ambiguous input produces a BLOCKED status artifact, not fabricated counts. Actual mounted image paths are checked under two common layouts; unresolved references can indicate archive-only or unrecognized layouts and are not automatically declared missing.

Local full-suite results: see evidence/step02b_tests.txt. Tests cover existing dataset/evaluator behavior plus unique/ambiguous mount discovery, missing input, actual CSV parsing on fixtures, malformed poses/assignments, repeated filenames across datasets, archive inspection without extraction, ZIP content exclusions, notebook parsing and input-boundary enforcement. No local test establishes official scorer parity or successful Kaggle execution.

Known limits: real mount/data permissions, threshold units/provenance, actual scene statistics, related datasets and near-duplicates remain unverified. Bounded inventory/archives may be incomplete; inspect completeness flags. Large archives are not extracted. Runtime output write probe must pass on Kaggle. Notebook UI labels were unavailable through the web reader, so the guide describes actions rather than claiming exact current button names.

Next human action: import notebook, attach ongoing competition, select CPU/internet off, execute, download imc_dataset_audit.zip and return it. Supervisor then reviews real evidence before approving a split or baseline. Stop here; Phase 3 is not started.
