# ToxD4C

Unified deep learning framework for multi‑task molecular toxicity prediction (Dual‑Driven Dynamic Deep Chemistry).

This README focuses on running, reproducing, and the ablation options currently supported in this repo.

## Highlights
- Hybrid GNN + Transformer encoder with optional 3D geometric and hierarchical branches
- Fingerprint module with attention fusion (ECFP/MACCS/RDK etc.)
- Supervised contrastive learning integrated into the training loss

## Architecture Overview
- Inputs: molecular graph (atoms/bonds), optional 3D coordinates, SMILES‑based fingerprints/descriptors.
- Hybrid encoder: GNN branch + Transformer branch with cross‑attention dynamic fusion (or concatenation).
- Optional encoders: Geometric message‑passing on distances (RBF/gaussian smearing); hierarchical multi‑level GCN.
- Fingerprint module: multiple classical fingerprints/descriptors with attention fusion into a single vector.
- Multi‑task head: classification and regression heads; tasks can be toggled for ablation or run as single‑endpoint.
- Representation learning: supervised contrastive loss (weighted by `contrastive_weight`).

High‑level data flow

```
SMILES / Graph / 3D coords / Fingerprints
           │           │           │
           │           │           └──► Fingerprint Module (ECFP/MACCS/etc., attention fusion)
           │           │
           │           └──► Geometric Encoder (optional)
           │
           └──► Hybrid Main Encoder
                 ├─ GNN Branch (GraphAttention or PyG GCN stack)
                 ├─ Transformer Branch
                 └─ Dynamic Fusion (cross‑attention) or Concatenation

          ──► Fused Graph Representation ──► Multi‑task Head (Cls + Reg)
                                         └─► (optional) SupCon representation for contrastive learning
```

Component relationships and switches
- GNN branch can be swapped via `--gnn_backbone {default,pyg_gcn_stack}`.
- Dynamic fusion can be disabled with `--disable_dynamic_fusion` (falls back to concatenation).
- Geometric, hierarchical and fingerprint branches can be independently disabled.
- Task routing: use all tasks (multi‑task), only classification/regression, or a single index (sensitivity runs).

## Quick Start
1) Prepare LMDB data with splits under a data directory (default below):
   - `train.lmdb`, `valid.lmdb`, `test.lmdb`
2) Train the full model (with contrastive learning):

```bash
python ToxD4C/train.py \
  --experiment_name "toxd4c_full_model" \
  --data_dir data/data/processed \
  --seed 42 --num_epochs 50 --batch_size 16 --deterministic
```

Outputs for each run are saved under `experiments/<name>_<timestamp>/`:
- `train.log`: run‑specific logs
- `checkpoints/<name>_best.pth`: best checkpoint
- `checkpoints/<name>_results.json`: summary metrics + config snapshot

## Key Flags
- `--disable_contrastive`: disable supervised contrastive learning (default is enabled)
- `--disable_gnn | --disable_transformer | --disable_geometric | --disable_hierarchical | --disable_fingerprint`: ablate components
- `--disable_dynamic_fusion` or `--fusion_method concatenation`: use concatenation instead of cross‑attention fusion
- `--gnn_backbone {default,pyg_gcn_stack}` and `--gcn_stack_layers N` (2–4): choose GNN backbone
- `--use_preprocessed` is enabled by default; preprocessed LMDB under `--preprocessed_dir` is used when present

### GNN backbone variants
- default (GraphAttentionNetwork): multi‑head attention message passing with configurable depth.
- pyg_gcn_stack (GCNStack): residual stack of PyG `GCNConv` layers with LayerNorm and dropout.
  - Recommended `--gcn_stack_layers` in [2, 4].
  - Works both in hybrid (with Transformer) and in GNN‑only ablations.

Example (GCN stack, hybrid):

```bash
python ToxD4C/train.py \
  --experiment_name "toxd4c_hybrid_gcnstack" \
  --gnn_backbone pyg_gcn_stack --gcn_stack_layers 3 \
  --seed 42 --num_epochs 50 --batch_size 16 --deterministic
```

## Common Ablations
All runs share the same base args as the full model; only the ablation flags are shown below.

- GNN only (baseline):
  `--disable_transformer --disable_geometric --disable_hierarchical --disable_fingerprint`
- GNN + Transformer (core):
  `--disable_geometric --disable_hierarchical --disable_fingerprint`
- GNN + Transformer + 3D:
  `--disable_hierarchical --disable_fingerprint`
- GNN + Transformer + Fingerprint:
  `--disable_geometric --disable_hierarchical`
- Full − Fingerprint:
  `--disable_fingerprint`
- Full − Contrastive:
  `--disable_contrastive`
- Concatenation Fusion:
  `--disable_dynamic_fusion`
- Classification only / Regression only:
  `--disable_regression` / `--disable_classification`

## Data & Preprocessing
- Training consumes LMDB splits (`train.lmdb`, `valid.lmdb`, `test.lmdb`).
- With `--use_preprocessed` (default), precomputed node/edge/3D tensors are read from `--preprocessed_dir`.
- If missing (or `--force_preprocess`), `preprocess_data.py` is invoked to cache tensors for faster training.

## Reproducibility & Splits
- Determinism: `--seed` and `--deterministic` set consistent training behavior and snapshot full run metadata.
- Splits: random / scaffold / cluster do not overlap by design; see `utils/splitter.py` for diagnostics.

## Notes on Recent Changes
- Contrastive learning is now part of the training objective, weighted by `config['contrastive_weight']` (default 0.3). To disable use `--disable_contrastive`.
- Logs are no longer written to a global file. Each run now logs to `experiments/<name>_<timestamp>/train.log`.

## Re‑running Old Experiments
Experiments trained before this change did not include the contrastive loss in the objective even if enabled in the config. For fair comparisons, re‑run experiments where `use_contrastive_learning: true` appears in the saved config.

Typical experiments to re‑run:
- `toxd4c_ablation_gnn_only`
- `toxd4c_ablation_gnn_transformer`
- `toxd4c_ablation_gnn_trans_3d`
- `toxd4c_ablation_gnn_trans_fp`
- `toxd4c_ablation_full_model` (ensure NOT passing `--disable_contrastive`)
- `toxd4c_ablation_full_no_fp`
- `toxd4c_ablation_concat_fusion`
- `toxd4c_ablation_classification_only`
- `toxd4c_ablation_regression_only`

Additionally, earlier `full_hybrid_gcnstack` runs that failed during forward have been fixed and should be re‑run.

## Inference
See `ToxD4C/inference_toxd4c.py` for batch inference on SMILES files using a trained checkpoint.

## License and Contributions
MIT‑licensed. Issues and PRs are welcome.
