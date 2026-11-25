# Phase 4C resource and cache

Use notebooks/step04c_haiper_capture_colab.ipynb in a fresh free Colab T4 without Drive. Run its five cells once in order. Upload imc_ns64_clean_runtime.zip in cell 1; cell 2 downloads and verifies the weights. Confirm DEPENDENCIES_PASS in cell 3. Upload imc2023_haiper_opaque.zip in cell 4. Enter available session seconds in cell 5. The driver caps its allowance at the remainder of 5400 seconds from cell 1 and reserves export time. Historical inference took 1087 seconds with one map; Phase 4C adds capture and replay.

Download phase4c_haiper_receipts_20261111.zip from /content and the SHA-256-named .db from /content/phase4c_haiper_run/20261111/capture_v1/. Use Colab Files if automatic downloads are interrupted. Preserve partial receipts and any completed parent. Avoid repeating matching. Real resource peaks and runtimes await those receipts.

Administrator commands from the project root, using the actual downloaded database filename and fresh output folders:

~~~powershell
python scripts/validate_ns64_geometry.py phase4c_haiper_receipts_20261111.zip ACTUAL_SHA256.db reports/phase4c_returned
python scripts/compare_ns64_mapper_parity.py phase4c_haiper_receipts_20261111.zip reports/phase4c_returned/validation.json reports/phase4c_parity
~~~