# Phase 3B supervisor handoff

**Registered organizer_baseline_v1_dev_full** on all 11 development datasets (1,563 images). Fold A scored 0.408975, Fold B scored 0.429336, and the equal-dataset mean was 0.418230 under both scoring variants. See [full report](../reports/phase3b_full/report.json) and [per-dataset metrics](../reports/phase3b_full/dataset_metrics.csv).

Batch 11 registered 225/225 images in one cluster across three reference scenes. Its mAA was 0.846065, clusterness 0.333333 and combined score 0.478247. Reconstruction took 7,886 seconds. The final session took 2 h 53 min and the prior timed-out session took 2 h 27 min. Across development, 1,435 cameras registered. Recorded session times including the timed-out attempt total at least 13 h 27 min, excluding export and any lost sessions. Peak worker RSS was 4.12 GiB and peak PyTorch allocation was 1.59 GiB. Use the archived four-hour batch 11 reconstruction allowance for paired comparisons. Supervisor review is next.

## Historical execution notes

Main artifacts: `notebooks/step03b_full_development_baseline.ipynb`, `scripts/build_phase3b_notebook.py`, `scripts/merge_baseline_batches.py`, `src/baselines/development/`, full configs/locks/development scoring metadata, recovered immutable smoke source, `tests/test_full_development.py`, and the six Phase 3B documents. Frozen benchmark files and Phase 3A notebook are unchanged. Full implementation identity is recorded in `full_implementation.json`.

## Run the first Kaggle batch

For unattended runs, use Kaggle **Save Version → Save & Run All** after preparing the loader. This runs a separate background session and saves its outputs with the notebook version. An interactive draft session can lose working files when its runtime resets. Downloaded local archives remain the reviewed evidence. See https://www.kaggle.com/docs/notebooks.

Before Save & Run All, keep only the Phase 3B loader as executable code in the pinned fork. Remove obsolete organizer inference cells and earlier batch loader cells from that working notebook so a top-to-bottom execution launches exactly the intended batch. Preserve its pinned environment and input attachments. The loader must initialize its imports, embedded source and settings within the same execution. After the saved run finishes, open that version's Output tab and download both report/cache archives. Check earlier saved versions for recoverable outputs before restarting a lost interactive run.

1. Use the same organizer notebook fork that passed the smoke test. Settings → Environment Preferences → **Pin to original environment**. Select **T4** accelerator, keep Internet **OFF**, and start a fresh session. Confirm `import sys; print(sys.version)` reports **3.10.12**.
2. Retain the ongoing competition, original wheel dataset, DINOv2 base/1, ALIKED n16/1 and LightGlue aliked/1 mounts described in `step03b_environment_and_resources.md`.
3. Upload the new `step03b_full_development_baseline.ipynb` file as a **private Kaggle dataset**, and attach that dataset to the working pinned fork. This preserves the known Python environment. Do not run the organizer's old inference cells as well.
4. Run this loader in a new code cell. It selects the new notebook, executes its source/setup cells, applies only session controls, then launches:

```python
from pathlib import Path
import json
matches = list(Path('/kaggle/input').rglob('step03b_full_development_baseline.ipynb'))
assert len(matches) == 1, f'Expected one uploaded Phase 3B notebook, found {matches}'
notebook = json.loads(matches[0].read_text())
code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
exec(compile(code[0], str(matches[0]), 'exec'), globals())
exec(compile(code[1], str(matches[0]), 'exec'), globals())

# Session controls only. Run batch 01 first; leave frozen model configuration alone.
settings.update(batches=['01'], export_id='01',
                work='/kaggle/working/phase3b_01',
                session_remaining_seconds=3600,
                export_cache=True, resume_archives=[])
SETTINGS.write_text(json.dumps(settings))
exec(compile(code[2], str(matches[0]), 'exec'), globals())
```

5. Download every `imc_baseline_dev_batch_01_<timestamp>.zip` and optional `_cache.zip` even if the run is PARTIAL/BLOCKED. The timestamp prevents overwriting previous evidence. Place the lightweight report ZIP in `D:\CV_Project` for local review. Do not submit predictions to Kaggle.
6. After review, run batches 02–11 in the fixed order from `step03b_execution_plan.md`, preferably one whole dataset per fresh session. Change `batches`, `export_id`, `work` and actual remaining seconds only. Larger batches may require up to the six-hour operational limit and still exceed the fixed SfM cap; report that bottleneck rather than altering the baseline.

## Resume

For an interrupted dataset, upload its last report and cache ZIPs as private inputs. Set `resume_archives` to their exact `/kaggle/input/.../*.zip` paths. Select the same batch and frozen model configuration. Completed stages are reused only if their content and context hashes match. A report alone can reuse a completed dataset; without feature/database cache, incomplete stages must rerun. Do not mix different attempts' conflicting files. Never manually mark a stage PASS.

Use separate working directories per batch so report archives do not repeat previously completed datasets. Import only one final completed report per dataset. Retain earlier partial archives separately as failure evidence; their paths can be supplied too provided they contain no duplicate completed results. The merger deliberately rejects duplicates rather than selecting a preferred score.

## Local validation and registration

From `D:\CV_Project`, provide explicit lightweight report paths (exclude cache ZIPs):

```powershell
python scripts/merge_baseline_batches.py imc_baseline_dev_batch_01_TIMESTAMP.zip --output reports/phase3b_review_01
```

Each output directory must be new. Partial reports list missing datasets and cannot generate a full combined score. After all eleven completed report paths are supplied:

```powershell
$reports = Get-ChildItem -LiteralPath reports/phase3b_final_inputs -Filter '*.zip' | Select-Object -ExpandProperty FullName
python scripts/merge_baseline_batches.py @reports --output reports/phase3b_full_verified --register
```

Put only the selected lightweight reports into `reports/phase3b_final_inputs` first. The command independently recomputes scores, validates every dataset, writes a new immutable baseline record, and creates `reports/phase3b_full_verified/imc_baseline_dev_full_report.zip`. Missing, conflicting or duplicate completed datasets block registration. Existing baseline records cannot be replaced.

Tests: **103 tests passed** using `python -m unittest discover -s tests -q`, including an end-to-end synthetic report merge. The suite covers original benchmark behavior and new source/config pins, membership, confirmation exclusion, cache corruption/resume, archives, exact coverage, all-unregistered failure scoring, equal-dataset aggregation, partial/full status, exclusive records and independent scoring. CPU tests do not establish that all eleven datasets fit GPU limits.

Supervisor readiness: ready for staged human Kaggle baseline execution. **Not ready for novel-model comparison or tuning.** Review complete baseline evidence first; keep confirmation locked throughout.
