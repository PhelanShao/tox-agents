# ToxD4C Reviewer Update Summary

This release integrates the code changes prepared in response to reviewer requests. Key updates:

## Reproducibility Controls (A0)
- Added `utils/reproducibility.py` with deterministic training utilities and run snapshot helpers.
- `train.py` exposes `--seed`/`--deterministic` flags and records environment metadata per experiment.

## Dataset Splitting & Analysis (A1)
- New `utils/splitter.py` provides scaffold, cluster, and external validation splits plus quality diagnostics.
- Training CLI now supports `--split_strategy`, automated preprocessing, resume/freeze helpers, and head-only fine-tuning.

## Uncertainty Quantification (A2)
- Added `utils/uncertainty.py` (temperature scaling, ensembles, conformal prediction, applicability domain).
- Inference outputs now include calibrated probabilities; training can enable/disable uncertainty flows.

## Architecture & Ablations (A3)
- Hybrid encoder accepts configurable backbones (`graph_attention` or `pyg_gcn_stack`) and dynamic fusion toggles.
- New transformer-only and residual GCN stack backbones for controlled ablation studies.
- Multi-task head supports single-endpoint and task disable switches, matching sensitivity experiments.

## Data Pipeline Improvements
- `data/lmdb_dataset.py` now consumes preprocessed LMDB tensors directly and handles directory-based databases.
- `preprocess_data.py` exposes CLI to build cached LMDB datasets (invoked automatically when needed).

## Documentation & Tooling
- README rewritten with focus on reproducibility, ablations, and run instructions.
- Added training/analysis helpers (e.g., `generate_scaffold_splits.py`, `analyze_endpoints.py`) retained locally for future automation.

Refer to individual module docstrings for reviewer tag references (A0–A3, R1C3, R2C11, etc.).
