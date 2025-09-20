#!/usr/bin/env python3
"""
Evaluate Tox21 predictions by aligning tox21_preds/*.csv with corresponding
label files under 'tox21 challenge' and computing Accuracy/F1/AUC per task.

Input conventions:
- Prediction CSVs are produced by tools/batch_infer_smiles.py or inference_toxd4c.py
  and contain columns: 'SMILES', '<Task>_prob' and '<Task>' (0/1) for 26 classification tasks.
- Label files are *.smiles with TAB-separated columns: SMILES, ID, label.

Usage example:
  python ToxD4C/tools/eval_tox21.py \
    --preds_dir tox21_preds \
    --labels_dir "tox21 challenge" \
    --output_csv tox21_eval_summary.csv
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
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
    # Ensure proper types
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    # Deduplicate on SMILES keeping first
    df = df.drop_duplicates(subset=['SMILES'], keep='first')
    return df


def find_pred_file(preds_dir: Path, stem: str) -> Optional[Path]:
    # Prefer <stem>_preds.csv
    candidate = preds_dir / f"{stem}_preds.csv"
    if candidate.exists():
        return candidate
    # Fallback: any csv containing the stem
    matches = list(preds_dir.glob(f"*{stem}*.csv"))
    if matches:
        # Choose the shortest name match for determinism
        matches.sort(key=lambda p: len(p.name))
        return matches[0]
    return None


def evaluate_one_task(pred_csv: Path, labels_df: pd.DataFrame, task: str) -> Tuple[Dict, Optional[pd.DataFrame]]:
    preds = pd.read_csv(pred_csv)
    if 'SMILES' not in preds.columns:
        return ({'error': f"Missing SMILES in {pred_csv.name}"}, None)

    # Match rows by SMILES
    df = preds.merge(labels_df[['SMILES', 'label']], on='SMILES', how='inner')
    n_total = len(labels_df)
    n_matched = len(df)

    if n_matched == 0:
        return ({'error': f"No overlap on SMILES for {pred_csv.name}"}, None)

    # Columns
    prob_col = f"{task}_prob"
    bin_col = task
    if bin_col not in df.columns:
        return ({'error': f"Missing column {bin_col} in {pred_csv.name}"}, None)

    y_true = df['label'].astype(int).to_numpy()
    y_pred = df[bin_col].astype(int).to_numpy()

    acc = accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

    auc = np.nan
    if prob_col in df.columns:
        try:
            auc = roc_auc_score(y_true, df[prob_col].to_numpy())
        except Exception:
            auc = np.nan

    return ({
        'task': task,
        'pred_file': pred_csv.name,
        'n_labels': int(n_total),
        'n_matched': int(n_matched),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc) if not np.isnan(auc) else None,
    }, df)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Tox21 predictions')
    parser.add_argument('--preds_dir', type=str, default='tox21_preds', help='Directory with prediction CSVs')
    parser.add_argument('--labels_dir', type=str, default='tox21 challenge', help='Directory with *.smiles labels')
    parser.add_argument('--output_csv', type=str, default='tox21_eval_summary.csv', help='Output summary CSV path')
    args = parser.parse_args()

    preds_dir = Path(args.preds_dir)
    labels_dir = Path(args.labels_dir)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    # Iterate over label files present
    for smiles_path in sorted(labels_dir.glob('*.smiles')):
        stem = smiles_path.stem.lower()
        task = STEM_TO_TASK.get(stem)
        if task is None:
            print(f"Skip (unmapped): {smiles_path.name}")
            continue

        pred_csv = find_pred_file(preds_dir, stem)
        if pred_csv is None:
            print(f"Missing predictions for {stem} ({task}) in {preds_dir}")
            continue

        labels_df = load_labels(smiles_path)
        metrics, _ = evaluate_one_task(pred_csv, labels_df, task)
        if 'error' in metrics:
            print(f"Error for {stem}: {metrics['error']}")
        else:
            print(f"{task}: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} auc={(metrics['auc'] if metrics['auc'] is not None else 'NA')}")
        results.append(metrics)

    if not results:
        print("No results to write.")
        return 1

    # Write summary CSV
    df_sum = pd.DataFrame(results)
    df_sum.to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path}")

    # Print macro averages over available tasks (only numeric columns)
    num_cols = ['accuracy', 'precision', 'recall', 'f1']
    avgs = {k: float(np.nanmean(df_sum[k])) for k in num_cols if k in df_sum.columns}
    if 'auc' in df_sum.columns:
        avgs['auc'] = float(np.nanmean(pd.to_numeric(df_sum['auc'], errors='coerce')))
    print("Macro averages:")
    for k, v in avgs.items():
        print(f"  {k}: {v:.3f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

