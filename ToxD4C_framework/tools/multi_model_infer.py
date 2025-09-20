#!/usr/bin/env python3
"""
Multi-model (ensemble) inference for SMILES files.

Loads multiple checkpoints, runs inference for each, and aggregates results via
- soft voting: average probabilities then apply threshold
- hard voting: majority vote on binary predictions

Writes one CSV per input *.smiles with ensemble columns.

Example
  python ToxD4C/tools/multi_model_infer.py \
    --model_paths \
      ToxD4C/experiments/toxd4c_baseline_complete_20250904_094335/checkpoints/toxd4c_baseline_complete_best.pth \
      ToxD4C/experiments/toxd4c_baseline_complete_20250903_223754/checkpoints/toxd4c_baseline_complete_best.pth \
      ToxD4C/experiments/toxd4c_baseline_complete_20250903_220804/checkpoints/toxd4c_baseline_complete_best.pth \
    --input_dir "tox21 challenge" --pattern "nr-er.smiles" \
    --output_dir tox21_preds --vote soft --prob_threshold 0.5 --device cuda
"""

import argparse
from pathlib import Path
from typing import List
import sys
import pandas as pd
import torch

# project imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from configs.toxd4c_config import CLASSIFICATION_TASKS, get_enhanced_toxd4c_config
from inference_toxd4c import ToxD4CPredictor, SmilesDataset
from data.lmdb_dataset import collate_lmdb_batch


def run_single(predictor: ToxD4CPredictor, smiles_file: Path, batch_size: int):
    smiles_list = smiles_file.read_text().splitlines()
    dataset = SmilesDataset(smiles_list)
    if len(dataset) == 0:
        return pd.DataFrame()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_lmdb_batch, num_workers=0, drop_last=False,
    )
    return predictor.predict_on_loader(loader)


def aggregate_ensemble(dfs: List[pd.DataFrame], vote: str = 'soft', prob_threshold: float = 0.5) -> pd.DataFrame:
    """Merge per-model prediction DataFrames safely and aggregate.

    We suffix every non-SMILES column with __m{i} to avoid name collisions, then
    average probabilities across models. Binary votes are majority if vote='hard'.
    """
    if not dfs:
        return pd.DataFrame()

    prepared: List[pd.DataFrame] = []
    for i, df in enumerate(dfs):
        df_i = df.copy()
        rename = {c: f"{c}__m{i}" for c in df_i.columns if c != 'SMILES'}
        df_i = df_i.rename(columns=rename)
        prepared.append(df_i)

    base = prepared[0]
    for df_i in prepared[1:]:
        base = base.merge(df_i, on='SMILES', how='inner')
    if base.empty:
        return base

    out = pd.DataFrame({'SMILES': base['SMILES']})

    # aggregate classification tasks only
    for task in CLASSIFICATION_TASKS:
        # find per-model prob columns like '<task>_prob__m0'
        prob_cols = [c for c in base.columns if c.startswith(f"{task}_prob__m")]
        if not prob_cols:
            # fallback: any column that contains task and 'prob'
            prob_cols = [c for c in base.columns if (task in c and 'prob' in c)]
        if not prob_cols:
            continue
        probs = base[prob_cols].to_numpy(dtype=float)
        prob_ens = probs.mean(axis=1)
        out[f'{task}_prob'] = prob_ens

        if vote == 'soft':
            out[task] = (prob_ens >= prob_threshold).astype(int)
        else:
            # majority vote on binary predictions '<task>__m<i>'
            bin_cols = [c for c in base.columns if c.startswith(f"{task}__m")]  # already 0/1 logits merged
            if bin_cols:
                bins = base[bin_cols].to_numpy(dtype=float)
                out[task] = (bins.mean(axis=1) >= 0.5).astype(int)
            else:
                out[task] = (prob_ens >= prob_threshold).astype(int)
    return out


def main():
    ap = argparse.ArgumentParser(description='Multi-model ensemble inference for SMILES files')
    ap.add_argument('--model_paths', nargs='+', type=str, required=True, help='List of checkpoint paths')
    ap.add_argument('--input_dir', type=str, default='tox21 challenge')
    ap.add_argument('--pattern', type=str, default='*.smiles')
    ap.add_argument('--output_dir', type=str, default='tox21_preds')
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--device', type=str, default=None, choices=['cpu','cuda'])
    ap.add_argument('--vote', type=str, default='soft', choices=['soft','hard'])
    ap.add_argument('--prob_threshold', type=float, default=0.5)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # device selection with fallback
    device = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

    # config (shared)
    config = get_enhanced_toxd4c_config()

    # load predictors
    predictors: List[ToxD4CPredictor] = []
    for mp in args.model_paths:
        try:
            predictors.append(ToxD4CPredictor(model_path=mp, config=config, device=device))
        except Exception as e:
            print(f"Failed to load model: {mp} — {e}")
    if not predictors:
        print('No valid models loaded. Exit.')
        return 1

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print(f'No files matched: {input_dir}/{args.pattern}')
        return 1

    for fp in files:
        print(f"\n-> Ensemble predicting: {fp}")
        model_dfs = []
        for pred in predictors:
            df = run_single(pred, fp, args.batch_size)
            if df.empty:
                print('   Warning: empty predictions for this model; skipping this model for this file')
                continue
            model_dfs.append(df)
        if not model_dfs:
            print('   Skipped: no model produced predictions')
            continue
        ens = aggregate_ensemble(model_dfs, vote=args.vote, prob_threshold=args.prob_threshold)
        if ens.empty:
            print('   Skipped: no common SMILES across models')
            continue
        out_path = output_dir / f"{fp.stem}_preds_ens.csv"
        ens.to_csv(out_path, index=False)
        print(f"   Saved ensemble: {out_path}  (n={len(ens)})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
