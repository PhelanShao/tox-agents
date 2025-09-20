#!/usr/bin/env python3
"""
Evaluate Tox21 predictions on a filtered subset that excludes exact duplicates
with the training set (non-overlap by InChIKey or canonical SMILES).

Inputs:
- --overlap_dir: directory created by tools/check_nonoverlap_and_similarity.py
  expects files: challenge_index.csv, direct_overlap.txt, challenge_nn_similarity.csv
- --preds_dir: directory with prediction CSVs (one per task), produced by batch_infer_smiles.py
- --labels_dir: directory with *.smiles label files (SMILES\tID\tlabel)

Outputs:
- Writes to --output_dir (created under tox21_overlap_check/ by default):
  - allowed_smiles.txt
  - no_overlap_eval_summary.csv
  - no_overlap_nn_similarity.csv (filtered)
  - no_overlap_similarity_stats.json
  - optional histogram: no_overlap_nn_similarity_hist.png
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
import pandas as pd
import numpy as np
import json

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


STEM_TO_TASK: Dict[str, str] = {
    # NR family
    'nr-ahr': 'NR-AhR',
    'nr-ar': 'NR-AR',
    'nr-ar-lbd': 'NR-AR-LBD',
    'nr-er': 'NR-ER',
    'nr-er-lbd': 'NR-ER-LBD',
    'nr-ppar-gamma': 'NR-PPAR-gamma',
    'nr-aromatase': 'NR-Aromatase',
    # SR family
    'sr-are': 'SR-ARE',
    'sr-atad5': 'SR-ATAD5',
    'sr-hse': 'SR-HSE',
    'sr-mmp': 'SR-MMP',
    'sr-p53': 'SR-p53',
}


def load_labels(smiles_path: Path) -> pd.DataFrame:
    df = pd.read_csv(smiles_path, sep='\t', header=None, names=['SMILES', 'ID', 'label'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df = df.drop_duplicates(subset=['SMILES'], keep='first')
    return df


def filter_allowed_smiles(overlap_dir: Path) -> pd.DataFrame:
    ch_idx = pd.read_csv(overlap_dir / 'challenge_index.csv')
    # Overlap identifiers can be InChIKeys or canonical SMILES depending on availability
    overlap_list = (overlap_dir / 'direct_overlap.txt').read_text().splitlines()
    overlap_set: Set[str] = set([s.strip() for s in overlap_list if s.strip()])

    # Allowed rows: inchikey not in overlap_set AND smiles_canonical not in overlap_set
    def is_overlapped(row) -> bool:
        ik = str(row['inchikey']) if not pd.isna(row.get('inchikey')) else None
        cs = str(row['smiles_canonical']) if not pd.isna(row.get('smiles_canonical')) else None
        if ik and ik in overlap_set:
            return True
        if cs and cs in overlap_set:
            return True
        return False

    ch_idx['overlap_flag'] = ch_idx.apply(is_overlapped, axis=1)
    allowed = ch_idx[~ch_idx['overlap_flag']].copy()
    return allowed[['smiles_raw', 'smiles_canonical', 'inchikey']]


def evaluate_on_allowed(pred_csv: Path, labels_df: pd.DataFrame, task: str, allowed_smiles: Set[str]) -> Dict:
    preds = pd.read_csv(pred_csv)
    if 'SMILES' not in preds.columns:
        return {'task': task, 'pred_file': pred_csv.name, 'error': 'Missing SMILES'}

    # Filter predictions first by allowed SMILES
    preds_f = preds[preds['SMILES'].isin(allowed_smiles)].copy()
    if preds_f.empty:
        return {'task': task, 'pred_file': pred_csv.name, 'n_allowed': 0, 'error': 'No allowed predictions'}

    # Join with labels
    df = preds_f.merge(labels_df[['SMILES', 'label']], on='SMILES', how='inner')
    if df.empty:
        return {'task': task, 'pred_file': pred_csv.name, 'n_allowed': len(preds_f), 'n_matched': 0, 'error': 'No label matches'}

    prob_col = f"{task}_prob"
    bin_col = task
    if bin_col not in df.columns:
        return {'task': task, 'pred_file': pred_csv.name, 'error': f'Missing column {bin_col}'}

    y_true = df['label'].astype(int).to_numpy()
    y_pred = df[bin_col].astype(int).to_numpy()
    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = None
    if prob_col in df.columns:
        try:
            auc = float(roc_auc_score(y_true, df[prob_col].to_numpy()))
        except Exception:
            auc = None

    return {
        'task': task,
        'pred_file': pred_csv.name,
        'n_allowed': int(len(preds_f)),
        'n_matched': int(len(df)),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(recall),
        'f1': float(f1),
        'auc': auc,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Tox21 on non-overlap subset (no exact duplicates)')
    parser.add_argument('--overlap_dir', type=str, default='tox21_overlap_check', help='Directory with overlap artifacts')
    parser.add_argument('--preds_dir', type=str, default='tox21_preds', help='Directory with prediction CSVs')
    parser.add_argument('--labels_dir', type=str, default='tox21 challenge', help='Directory with *.smiles labels')
    parser.add_argument('--output_dir', type=str, default='tox21_overlap_check/no_overlap_eval', help='Output directory')
    args = parser.parse_args()

    overlap_dir = Path(args.overlap_dir)
    preds_dir = Path(args.preds_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allowed SMILES (raw) based on non-overlap
    allowed_df = filter_allowed_smiles(overlap_dir)
    allowed_smiles: Set[str] = set(allowed_df['smiles_raw'].tolist())
    (out_dir / 'allowed_smiles.txt').write_text('\n'.join(sorted(allowed_smiles)))
    print(f"Allowed (non-overlap) molecules: {len(allowed_smiles)}")

    results = []
    for smiles_path in sorted(labels_dir.glob('*.smiles')):
        stem = smiles_path.stem.lower()
        task = STEM_TO_TASK.get(stem)
        if task is None:
            print(f"Skip (unmapped): {smiles_path.name}")
            continue

        # Prediction CSV
        pred_csv = preds_dir / f"{stem}_preds.csv"
        if not pred_csv.exists():
            # Fallback: any CSV containing stem
            matches = list(preds_dir.glob(f"*{stem}*.csv"))
            if not matches:
                print(f"Missing predictions for {stem} ({task})")
                continue
            matches.sort(key=lambda p: len(p.name))
            pred_csv = matches[0]

        labels_df = load_labels(smiles_path)
        metrics = evaluate_on_allowed(pred_csv, labels_df, task, allowed_smiles)
        if 'error' in metrics:
            print(f"{task}: {metrics['error']}")
        else:
            print(f"{task}: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auc={(metrics['auc'] if metrics['auc'] is not None else 'NA')}  n={metrics['n_matched']}")
        results.append(metrics)

    # Save summary
    sum_df = pd.DataFrame(results)
    sum_df.to_csv(out_dir / 'no_overlap_eval_summary.csv', index=False)
    print(f"Saved: {out_dir / 'no_overlap_eval_summary.csv'}")

    # Filter similarity file if present
    nn_path = overlap_dir / 'challenge_nn_similarity.csv'
    if nn_path.exists():
        nn_df = pd.read_csv(nn_path)
        # Join by inchikey if available
        allowed_ikeys = set(allowed_df['inchikey'].dropna().tolist())
        if allowed_ikeys:
            nn_f = nn_df[nn_df['inchikey'].isin(allowed_ikeys)].copy()
        else:
            # Fallback: by canonical smiles
            allowed_canos = set(allowed_df['smiles_canonical'].tolist())
            nn_f = nn_df[nn_df['smiles_canonical'].isin(allowed_canos)].copy()

        nn_f.to_csv(out_dir / 'no_overlap_nn_similarity.csv', index=False)
        # Stats
        sims = nn_f['nn_sim'].dropna().to_numpy()
        dist = {
            'n': int(sims.size),
            'mean': float(np.mean(sims)) if sims.size else None,
            'median': float(np.median(sims)) if sims.size else None,
            'p95': float(np.percentile(sims, 95)) if sims.size else None,
            'p99': float(np.percentile(sims, 99)) if sims.size else None,
            'max': float(np.max(sims)) if sims.size else None,
        }
        (out_dir / 'no_overlap_similarity_stats.json').write_text(json.dumps(dist, indent=2))
        print(f"Saved: {out_dir / 'no_overlap_similarity_stats.json'}")

        # Optional histogram
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns  # optional
            plt.figure(figsize=(6,4))
            plt.hist(sims, bins=50, color='#4C78A8', alpha=0.85)
            plt.xlabel('ECFP4 Tanimoto (NN)')
            plt.ylabel('Count')
            plt.title('NN Similarity Distribution (Non-overlap subset)')
            plt.tight_layout()
            plt.savefig(out_dir / 'no_overlap_nn_similarity_hist.png', dpi=200)
            plt.close()
            print(f"Saved: {out_dir / 'no_overlap_nn_similarity_hist.png'}")
        except Exception as e:
            print(f"Skip histogram (matplotlib not available?): {e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

