#!/usr/bin/env python3
"""
Export per-endpoint Top-K (lowest/highest NN similarity) molecules with
predicted probabilities and ground-truth labels, and compute small stats.

Inputs
- overlap_dir: contains challenge_nn_similarity.csv (or no_overlap_eval/no_overlap_nn_similarity.csv)
- preds_dir: prediction CSVs (e.g., nr-ahr_preds.csv) with columns SMILES, <Task>_prob, <Task>
- labels_dir: tox21 *.smiles with columns SMILES\tID\tlabel

Outputs under output_dir/<task>/<mode>/
- topk_stats.csv: SMILES, canonical, label, prob, pred, nn_sim, nn_train_smiles, correct
- summary.json: small metrics for the Top-K subset
- summary_all.csv: aggregated per task across modes (written at output_dir root)
"""

import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import json
from rdkit import Chem


STEM_TO_TASK: Dict[str, str] = {
    'nr-ahr': 'NR-AhR',
    'nr-ar': 'NR-AR',
    'nr-ar-lbd': 'NR-AR-LBD',
    'nr-er': 'NR-ER',
    'nr-er-lbd': 'NR-ER-LBD',
    'nr-ppar-gamma': 'NR-PPAR-gamma',
    'nr-aromatase': 'NR-Aromatase',
    'sr-are': 'SR-ARE',
    'sr-atad5': 'SR-ATAD5',
    'sr-hse': 'SR-HSE',
    'sr-mmp': 'SR-MMP',
    'sr-p53': 'SR-p53',
}


def canon(s: str) -> str:
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, canonical=True) if m is not None else None
    except Exception:
        return None


def load_nn_df(overlap_dir: Path) -> pd.DataFrame:
    p = overlap_dir / 'no_overlap_eval' / 'no_overlap_nn_similarity.csv'
    if not p.exists():
        p = overlap_dir / 'challenge_nn_similarity.csv'
    if not p.exists():
        raise FileNotFoundError('NN similarity table not found under overlap_dir')
    df = pd.read_csv(p)
    if 'smiles_canonical' not in df.columns or 'nn_sim' not in df.columns:
        raise ValueError('NN table missing required columns')
    return df[['smiles_canonical', 'nn_sim', 'nn_train_smiles']]


def load_labels_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=None, names=['SMILES', 'ID', 'label'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df['label'] = df['label'].fillna(0).astype(int)
    df = df.drop_duplicates(subset=['SMILES'], keep='first')
    return df[['SMILES', 'label']]


def find_labels(labels_dir: Path, stem: str) -> pd.DataFrame:
    exact = labels_dir / f'{stem}.smiles'
    if exact.exists():
        return load_labels_file(exact)
    # fuzzy fallback: unique match containing stem
    cands = list(labels_dir.glob(f'*{stem}*.smiles'))
    if len(cands) == 1:
        return load_labels_file(cands[0])
    # otherwise, no safe match
    return None


def process_task(stem: str, task: str, nn_df: pd.DataFrame, preds_dir: Path, labels_dir: Path,
                 out_root: Path, top_k: int, mode: str) -> Dict:
    # locate preds file
    pred_path = preds_dir / f'{stem}_preds.csv'
    if not pred_path.exists():
        matches = list(preds_dir.glob(f'*{stem}*.csv'))
        if not matches:
            return {'task': task, 'mode': mode, 'exported': 0, 'reason': 'preds_missing'}
        matches.sort(key=lambda p: len(p.name))
        pred_path = matches[0]
    preds = pd.read_csv(pred_path)
    if 'SMILES' not in preds.columns:
        return {'task': task, 'mode': mode, 'exported': 0, 'reason': 'no_SMILES_in_preds'}
    prob_col = f'{task}_prob'
    bin_col = task
    if prob_col not in preds.columns or bin_col not in preds.columns:
        return {'task': task, 'mode': mode, 'exported': 0, 'reason': 'prob/bin cols missing'}

    # add canonical
    preds['canonical'] = preds['SMILES'].map(canon)
    # labels
    labs = find_labels(labels_dir, stem)

    df = preds.merge(nn_df, left_on='canonical', right_on='smiles_canonical', how='inner')
    if labs is not None:
        df = df.merge(labs, on='SMILES', how='left')
    if df.empty:
        return {'task': task, 'mode': mode, 'exported': 0, 'reason': 'no_overlap_between_preds_and_nn'}

    asc = True if mode == 'low' else False
    df = df.sort_values('nn_sim', ascending=asc).head(top_k).copy()
    if df.empty:
        return {'task': task, 'mode': mode, 'exported': 0}

    out_dir = out_root / stem / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    # compute correctness
    if 'label' in df.columns:
        df['correct'] = (df[bin_col].astype(int) == df['label'].astype(int)).astype(int)
    else:
        df['label'] = np.nan
        df['correct'] = np.nan

    # filename stub consistent with visualize script: <mode>_<index:04d>_sim_<nn_sim:.3f>
    # Using current DataFrame index to build a stable stub for this export
    stubs = [f"{mode}_{int(i):04d}_sim_{row.nn_sim:.3f}" for i, row in df.iterrows()]
    df['file_stub'] = stubs

    export_cols = ['file_stub', 'SMILES', 'canonical', 'label', prob_col, bin_col, 'nn_sim', 'nn_train_smiles', 'correct']
    present_cols = [c for c in export_cols if c in df.columns]
    df[present_cols].to_csv(out_dir / 'topk_stats.csv', index=False)

    # summary metrics
    prob = df[prob_col].to_numpy()
    metrics = {
        'task': task,
        'mode': mode,
        'n': int(len(df)),
        'prob_mean': float(np.mean(prob)),
        'prob_median': float(np.median(prob)),
        'prob_min': float(np.min(prob)),
        'prob_max': float(np.max(prob)),
        'nn_sim_mean': float(np.mean(df['nn_sim'])),
        'nn_sim_min': float(np.min(df['nn_sim'])),
        'nn_sim_max': float(np.max(df['nn_sim'])),
    }
    if 'label' in df.columns and df['label'].notna().any():
        acc = float(np.mean((df[bin_col].astype(int) == df['label'].astype(int)).to_numpy()))
        metrics['accuracy'] = acc
        metrics['positives'] = int(df['label'].sum())
    (out_dir / 'summary.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description='Export Top-K stats (pred prob + label) per tox21 endpoint based on NN similarity')
    ap.add_argument('--overlap_dir', type=str, default='tox21_overlap_check')
    ap.add_argument('--preds_dir', type=str, default='tox21_preds')
    ap.add_argument('--labels_dir', type=str, default='tox21 challenge')
    ap.add_argument('--output_dir', type=str, default='tox21_overlap_check/topk_stats')
    ap.add_argument('--top_k', type=int, default=5)
    ap.add_argument('--mode', type=str, default='low', choices=['low','high','both'])
    ap.add_argument('--tasks', type=str, default='all', help='Comma-separated stems or "all"')
    args = ap.parse_args()

    overlap_dir = Path(args.overlap_dir)
    preds_dir = Path(args.preds_dir)
    labels_dir = Path(args.labels_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    nn_df = load_nn_df(overlap_dir)

    if args.tasks.strip().lower() == 'all':
        stems = sorted(STEM_TO_TASK.keys())
    else:
        stems = [s.strip().lower() for s in args.tasks.split(',') if s.strip()]

    modes = ['low','high'] if args.mode == 'both' else [args.mode]
    rows: List[Dict] = []
    for stem in stems:
        task = STEM_TO_TASK.get(stem, None)
        if task is None:
            continue
        for mode in modes:
            m = process_task(stem, task, nn_df, preds_dir, labels_dir, out_root, args.top_k, mode)
            rows.append(m)

    pd.DataFrame(rows).to_csv(out_root / 'summary_all.csv', index=False)
    print(f'Done. Results at: {out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
