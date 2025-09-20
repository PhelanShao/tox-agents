#!/usr/bin/env python3
"""
Plot 2x2 confusion-matrix heatmaps (Pred 0/1 vs Actual 0/1) for all Tox21 endpoints.

Features
- Reads predictions (<stem>_preds.csv) and labels (<stem>.smiles)
- Thresholds probabilities at --prob_threshold (default 0.5)
- Optional: restrict to non-overlap subset using overlap_dir/no_overlap_eval/allowed_smiles.txt
- Saves per-endpoint PNG heatmaps and a summary CSV with counts and metrics

Example
  python ToxD4C/tools/plot_confusion_heatmaps.py \
    --preds_dir tox21_preds \
    --labels_dir "tox21 challenge" \
    --output_dir tox21_overlap_check/confusion_heatmaps \
    --prob_threshold 0.5 \
    --use_non_overlap --overlap_dir tox21_overlap_check
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def load_labels(labels_dir: Path, stem: str) -> Optional[pd.DataFrame]:
    exact = labels_dir / f"{stem}.smiles"
    if exact.exists():
        df = pd.read_csv(exact, sep='\t', header=None, names=['SMILES','ID','label'])
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        return df[['SMILES','label']].drop_duplicates('SMILES', keep='first')
    cands = list(labels_dir.glob(f"*{stem}*.smiles"))
    if len(cands) == 1:
        df = pd.read_csv(cands[0], sep='\t', header=None, names=['SMILES','ID','label'])
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        return df[['SMILES','label']].drop_duplicates('SMILES', keep='first')
    return None


def load_allowed_smiles(overlap_dir: Path) -> Optional[set]:
    p = overlap_dir / 'no_overlap_eval' / 'allowed_smiles.txt'
    if p.exists():
        return set(s.strip() for s in p.read_text().splitlines() if s.strip())
    return None


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    # TN, FP, FN, TP
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def plot_confusion(task: str, tn: int, fp: int, fn: int, tp: int, N: int, out_png: Path, title_suffix: str = ""):
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['axes.unicode_minus'] = False
    cm = np.array([[tn, fp],[fn, tp]], dtype=float)
    # normalized by N for annotation
    perc = cm / max(1, N)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred 0', 'Pred 1'])
    ax.set_yticklabels(['Actual 0', 'Actual 1'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i,j])}\n({perc[i,j]*100:.1f}%)", va='center', ha='center', color='#17375E', fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Count')
    title = f"{task} Confusion Matrix{title_suffix}  (N={N})"
    ax.set_title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Plot confusion-matrix heatmaps for all endpoints')
    ap.add_argument('--preds_dir', type=str, default='tox21_preds')
    ap.add_argument('--labels_dir', type=str, default='tox21 challenge')
    ap.add_argument('--output_dir', type=str, default='tox21_overlap_check/confusion_heatmaps')
    ap.add_argument('--prob_threshold', type=float, default=0.5)
    ap.add_argument('--use_non_overlap', action='store_true', help='Filter to non-overlap subset if allowed_smiles.txt is present')
    ap.add_argument('--overlap_dir', type=str, default='tox21_overlap_check')
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed = load_allowed_smiles(Path(args.overlap_dir)) if args.use_non_overlap else None

    rows = []
    for stem, task in STEM_TO_TASK.items():
        pred_path = preds_dir / f'{stem}_preds.csv'
        if not pred_path.exists():
            # try fuzzy
            m = list(preds_dir.glob(f'*{stem}*.csv'))
            if not m:
                rows.append({'task': task, 'status': 'missing_preds'})
                continue
            m.sort(key=lambda p: len(p.name))
            pred_path = m[0]
        preds = pd.read_csv(pred_path)
        if 'SMILES' not in preds.columns:
            rows.append({'task': task, 'status': 'no_SMILES_column'})
            continue
        prob_col = f'{task}_prob'
        if prob_col not in preds.columns:
            rows.append({'task': task, 'status': f'missing {prob_col}'})
            continue
        preds = preds[['SMILES', prob_col]].copy()
        preds['pred_bin'] = (preds[prob_col] >= args.prob_threshold).astype(int)

        labs = load_labels(labels_dir, stem)
        if labs is None:
            rows.append({'task': task, 'status': 'missing_labels'})
            continue
        df = preds.merge(labs, on='SMILES', how='inner')
        if df.empty:
            rows.append({'task': task, 'status': 'no_match'})
            continue
        if allowed is not None:
            df = df[df['SMILES'].isin(allowed)].copy()
            title_suffix = ' (non-overlap)'
        else:
            title_suffix = ''
        if df.empty:
            rows.append({'task': task, 'status': 'empty_after_filter'})
            continue

        y_true = df['label'].astype(int).to_numpy()
        y_pred = df['pred_bin'].astype(int).to_numpy()
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        N = int(len(df))
        out_png = out_dir / f'confusion_{stem}.png'
        plot_confusion(task, tn, fp, fn, tp, N, out_png, title_suffix)

        acc = float(((y_true == y_pred).sum()) / N)
        prec = float(tp / max(1, tp + fp))
        rec = float(tp / max(1, tp + fn))
        rows.append({'task': task, 'N': N, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'accuracy': acc, 'precision': prec, 'recall': rec, 'status': 'ok'})

    pd.DataFrame(rows).to_csv(out_dir / 'confusion_summary.csv', index=False)
    print(f'Done. Heatmaps at: {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

