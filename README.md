# Image Matching Challenge research

Research code, experiment protocols, and audit records for an Image Matching Challenge reconstruction pipeline. Start with [AGENTS.md](AGENTS.md) and the [frozen evaluation protocol](docs/phase4i_evaluation_protocol.md).

## Layout

- `src/` contains scientific methods, baseline implementations, and checkpoint contracts.
- `scripts/`, `notebooks/`, and `tests/` contain runners and regression fixtures.
- `configs/` pins the benchmark, model, evaluator, and execution settings.
- `experiments/` and `reports/` contain experiment records and verification receipts.
- `snapshots/` records upstream source snapshots used by the project.
- `docs/` describes the phase decisions and operator workflows.

## Continue on another PC

Use Python 3.13 for the Windows CPU test workflow and the pinned Python 3.11.11/PyTorch 2.5.1+cu121 runtime for Dioscuri Kaggle stages. See the [handoff](docs/WORKSTATION_HANDOFF.md) for artifact transfer and current status. Keep the frozen imc2025_local_v1 split, evaluator, and comparators for all reported comparisons.
