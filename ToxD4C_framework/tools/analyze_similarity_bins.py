#!/usr/bin/env python3
"""
Analyze performance across molecular similarity bins for the Tox21 non-overlap subset.

Inputs
- --overlap_dir: directory produced by check_nonoverlap_and_similarity.py
  Requires: challenge_index.csv, direct_overlap.txt, and no_overlap_eval/no_overlap_nn_similarity.csv
- --preds_dir: directory with predictions CSVs (contains <task> and <task>_prob)
- --labels_dir: directory with Tox21 *.smiles label files (SMILES\tID\tlabel)

Outputs (under --output_dir)
- similarity_bin_metrics_fixed.csv: metrics per task and per fixed bin
- similarity_bin_metrics_percentile.csv: metrics per task and per percentile bin
- similarity_bins_fixed.png: bar plots (AUC and Accuracy) over fixed bins (Arial font)
- similarity_bins_percentile.png: bar plots over percentile bins
 - similarity_scatter_metric_pretty.png: pretty discrete scatter for a chosen metric (default ROC-AUC)
 - similarity_metric_summary.json: summary stats for the chosen metric over rolling windows
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import json

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

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


def gaussian_smooth(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, bw: float = 0.08, min_points: int = 10) -> np.ndarray:
    """Simple Nadaraya–Watson kernel regression with Gaussian kernel.
    Returns smoothed y on x_grid. If effective weight < min_points, returns NaN.
    """
    y_sm = np.full_like(x_grid, np.nan, dtype=float)
    for i, xv in enumerate(x_grid):
        w = np.exp(-0.5 * ((x - xv) / max(1e-6, bw)) ** 2)
        sw = w.sum()
        if sw >= min_points:
            y_sm[i] = float((w * y).sum() / sw)
    return y_sm


def load_allowed_map(overlap_dir: Path) -> pd.DataFrame:
    # Allowed mapping from raw SMILES to canonical/ikey
    ch_idx = pd.read_csv(overlap_dir / 'challenge_index.csv')
    overlap = set((overlap_dir / 'direct_overlap.txt').read_text().splitlines())

    def is_overlap(row) -> bool:
        ik = str(row['inchikey']) if not pd.isna(row.get('inchikey')) else None
        cs = str(row['smiles_canonical']) if not pd.isna(row.get('smiles_canonical')) else None
        return (ik in overlap) or (cs in overlap)

    ch_idx['overlap_flag'] = ch_idx.apply(is_overlap, axis=1)
    allowed = ch_idx[~ch_idx['overlap_flag']].copy()

    # Similarity table (already filtered for non-overlap by eval script)
    nn_path = overlap_dir / 'no_overlap_eval' / 'no_overlap_nn_similarity.csv'
    if not nn_path.exists():
        # Fallback to the unfiltered NN file (then merge and drop)
        nn_path = overlap_dir / 'challenge_nn_similarity.csv'
    nn_df = pd.read_csv(nn_path)

    # Merge by inchikey if available else by canonical smiles
    if 'inchikey' in nn_df.columns and allowed['inchikey'].notna().any():
        merged = allowed.merge(nn_df[['inchikey', 'smiles_canonical', 'nn_sim']], on=['inchikey', 'smiles_canonical'], how='left')
    else:
        merged = allowed.merge(nn_df[['smiles_canonical', 'nn_sim']], on='smiles_canonical', how='left')

    merged = merged[['smiles_raw', 'smiles_canonical', 'inchikey', 'nn_sim']].drop_duplicates('smiles_raw')
    return merged


def load_labels(smiles_path: Path) -> pd.DataFrame:
    df = pd.read_csv(smiles_path, sep='\t', header=None, names=['SMILES', 'ID', 'label'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    df = df.drop_duplicates(subset=['SMILES'], keep='first')
    return df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': auc,
    }


def eval_bins(df: pd.DataFrame, task: str, bins: List[float], labels: List[str]) -> pd.DataFrame:
    rows = []
    for b_lo, b_hi, label in zip(bins[:-1], bins[1:], labels):
        m = (df['nn_sim'] >= b_lo) & (df['nn_sim'] < b_hi if b_hi < 1.0 else df['nn_sim'] <= 1.0)
        part = df[m]
        if part.empty:
            rows.append({'task': task, 'bin': label, 'n': 0, 'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'auc': None})
            continue
        y_true = part['label'].astype(int).to_numpy()
        y_pred = part[task].astype(int).to_numpy() if task in part.columns else (part[f'{task}'] > 0.5).astype(int).to_numpy()
        y_prob = part.get(f'{task}_prob', None)
        y_prob = y_prob.to_numpy() if y_prob is not None else None
        met = compute_metrics(y_true, y_pred, y_prob)
        rows.append({'task': task, 'bin': label, 'n': int(len(part)), **met})
    return pd.DataFrame(rows)


def rolling_eval(df: pd.DataFrame, task: str, window: int, step: int) -> pd.DataFrame:
    df2 = df[['nn_sim', 'label', task, f'{task}_prob']].dropna(subset=['nn_sim']).copy()
    df2 = df2.sort_values('nn_sim').reset_index(drop=True)
    n = len(df2)
    rows = []
    if n < window:
        return pd.DataFrame(rows)
    for start in range(0, n - window + 1, step):
        part = df2.iloc[start:start+window]
        y_true = part['label'].astype(int).to_numpy()
        y_pred = part[task].astype(int).to_numpy()
        y_prob = part[f'{task}_prob'].to_numpy() if f'{task}_prob' in part.columns else None
        met = compute_metrics(y_true, y_pred, y_prob)
        x = float(part['nn_sim'].mean())
        rows.append({'task': task, 'x': x, 'n': int(len(part)), **met})
    return pd.DataFrame(rows)


def run_analysis(overlap_dir: Path, preds_dir: Path, labels_dir: Path, output_dir: Path,
                 fixed_bins: List[float], percentiles: List[float],
                 rolling_window: int = 0, rolling_step: int = 0,
                 style_nature: bool = False, point_size: int = 28, alpha: float = 0.6,
                 cmap: str = 'viridis', smooth_window: int = 25,
                 scatter_metric: str = 'auc',
                 xlim: str = '0,1', ylim_auc: str = '0.5,1.0', ylim_acc: str = '0.5,1.0') -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Allowed mapping with similarity
    allow_map = load_allowed_map(overlap_dir)
    # Global similarity for percentile computation (drop NaNs)
    sims = allow_map['nn_sim'].dropna().to_numpy()

    # Prepare bin specs
    fixed_labels = [f"{fixed_bins[i]:.1f}-{fixed_bins[i+1]:.1f}" for i in range(len(fixed_bins)-1)]
    perc_edges = np.percentile(sims, percentiles)
    perc_labels = [f"P{int(percentiles[i])}-P{int(percentiles[i+1])}" for i in range(len(percentiles)-1)]

    all_fixed = []
    all_perc = []
    all_roll = []

    # Iterate tasks
    for stem, task in STEM_TO_TASK.items():
        pred_csv = preds_dir / f"{stem}_preds.csv"
        if not pred_csv.exists():
            matches = list(preds_dir.glob(f"*{stem}*.csv"))
            if not matches:
                continue
            matches.sort(key=lambda p: len(p.name))
            pred_csv = matches[0]

        lab_path = labels_dir / f"{stem}.smiles"
        if not lab_path.exists():
            # Some files may use hyphens/underscores variations; skip if missing
            continue

        preds = pd.read_csv(pred_csv)
        labels = load_labels(lab_path)

        # Join and filter to allowed
        df = preds.merge(labels[['SMILES', 'label']], on='SMILES', how='inner')
        df = df.merge(allow_map[['smiles_raw', 'nn_sim']], left_on='SMILES', right_on='smiles_raw', how='inner')

        # Fixed bins
        fx_df = eval_bins(df, task, fixed_bins, fixed_labels)
        all_fixed.append(fx_df)

        # Percentile bins
        pc_df = eval_bins(df, task, perc_edges.tolist(), perc_labels)
        all_perc.append(pc_df)

        # Rolling window (discrete scatter, no bins)
        if rolling_window and rolling_window > 1:
            step = rolling_step if rolling_step and rolling_step > 0 else max(1, rolling_window // 5)
            r_df = rolling_eval(df, task, rolling_window, step)
            if not r_df.empty:
                all_roll.append(r_df)

    # Concatenate and save
    if all_fixed:
        fixed_out = pd.concat(all_fixed, ignore_index=True)
        fixed_out.to_csv(output_dir / 'similarity_bin_metrics_fixed.csv', index=False)
    if all_perc:
        perc_out = pd.concat(all_perc, ignore_index=True)
        perc_out.to_csv(output_dir / 'similarity_bin_metrics_percentile.csv', index=False)

    # Plots (overall macro averages)
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.sans-serif'] = ['Arial']
        mpl.rcParams['axes.unicode_minus'] = False
        if style_nature:
            # A clean, high-contrast style reminiscent of Nature figures
            mpl.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': '#262626',
                'axes.linewidth': 1.0,
                'grid.color': '#E5E5E5',
                'grid.linestyle': '-',
                'grid.linewidth': 0.8,
                'axes.grid': False,
                'savefig.dpi': 300,
            })

        if all_fixed:
            fixed_avg = fixed_out.groupby('bin').agg({'accuracy': 'mean', 'auc': 'mean', 'n': 'sum'}).reset_index()
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].bar(fixed_avg['bin'], fixed_avg['auc'], color='#4C78A8')
            ax[0].set_title('AUC vs Similarity (fixed bins)')
            ax[0].set_ylabel('AUC')
            ax[0].set_xlabel('ECFP4 Tanimoto bin')
            ax[0].set_ylim(0.0, 1.0)

            ax[1].bar(fixed_avg['bin'], fixed_avg['accuracy'], color='#72B7B2')
            ax[1].set_title('Accuracy vs Similarity (fixed bins)')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_xlabel('ECFP4 Tanimoto bin')
            ax[1].set_ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(output_dir / 'similarity_bins_fixed.png', dpi=200, bbox_inches='tight')
            plt.close()

            # Scatter by task with bin midpoints
            # Build midpoints for x-axis
            mid_map = {fixed_labels[i]: (fixed_bins[i] + fixed_bins[i+1]) / 2.0 for i in range(len(fixed_labels))}
            tmp = fixed_out.dropna(subset=['auc', 'accuracy']).copy()
            tmp['x'] = tmp['bin'].map(mid_map)

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(tmp['x'], tmp['auc'], s=16, alpha=0.35, color='#4C78A8', edgecolors='none')
            mean_auc = tmp.groupby('x')['auc'].mean().reset_index()
            ax[0].plot(mean_auc['x'], mean_auc['auc'], color='#1F4E79', linewidth=2)
            ax[0].set_title('AUC vs NN Similarity (scatter, fixed bins)')
            ax[0].set_xlabel('ECFP4 Tanimoto (bin midpoint)')
            ax[0].set_ylabel('AUC')
            ax[0].set_ylim(0.0, 1.0)

            ax[1].scatter(tmp['x'], tmp['accuracy'], s=16, alpha=0.35, color='#72B7B2', edgecolors='none')
            mean_acc = tmp.groupby('x')['accuracy'].mean().reset_index()
            ax[1].plot(mean_acc['x'], mean_acc['accuracy'], color='#2D6A6D', linewidth=2)
            ax[1].set_title('Accuracy vs NN Similarity (scatter, fixed bins)')
            ax[1].set_xlabel('ECFP4 Tanimoto (bin midpoint)')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(output_dir / 'similarity_scatter_fixed.png', dpi=200, bbox_inches='tight')
            plt.close()

        if all_perc:
            perc_avg = perc_out.groupby('bin').agg({'accuracy': 'mean', 'auc': 'mean', 'n': 'sum'}).reset_index()
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].bar(perc_avg['bin'], perc_avg['auc'], color='#4C78A8')
            ax[0].set_title('AUC vs Similarity (percentile bins)')
            ax[0].set_ylabel('AUC')
            ax[0].set_xlabel('ECFP4 Tanimoto percentile bin')
            ax[0].set_ylim(0.0, 1.0)

            ax[1].bar(perc_avg['bin'], perc_avg['accuracy'], color='#72B7B2')
            ax[1].set_title('Accuracy vs Similarity (percentile bins)')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_xlabel('ECFP4 Tanimoto percentile bin')
            ax[1].set_ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(output_dir / 'similarity_bins_percentile.png', dpi=200, bbox_inches='tight')
            plt.close()

            # Scatter for percentile bins (x=bin midpoint in similarity space)
            p_mid_map = {perc_labels[i]: (perc_edges[i] + perc_edges[i+1]) / 2.0 for i in range(len(perc_labels))}
            tmp = perc_out.dropna(subset=['auc', 'accuracy']).copy()
            tmp['x'] = tmp['bin'].map(p_mid_map)

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(tmp['x'], tmp['auc'], s=16, alpha=0.35, color='#4C78A8', edgecolors='none')
            mean_auc = tmp.groupby('x')['auc'].mean().reset_index()
            ax[0].plot(mean_auc['x'], mean_auc['auc'], color='#1F4E79', linewidth=2)
            ax[0].set_title('AUC vs NN Similarity (scatter, percentile bins)')
            ax[0].set_xlabel('ECFP4 Tanimoto (bin midpoint)')
            ax[0].set_ylabel('AUC')
            ax[0].set_ylim(0.0, 1.0)

            ax[1].scatter(tmp['x'], tmp['accuracy'], s=16, alpha=0.35, color='#72B7B2', edgecolors='none')
            mean_acc = tmp.groupby('x')['accuracy'].mean().reset_index()
            ax[1].plot(mean_acc['x'], mean_acc['accuracy'], color='#2D6A6D', linewidth=2)
            ax[1].set_title('Accuracy vs NN Similarity (scatter, percentile bins)')
            ax[1].set_xlabel('ECFP4 Tanimoto (bin midpoint)')
            ax[1].set_ylabel('Accuracy')
            ax[1].set_ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(output_dir / 'similarity_scatter_percentile.png', dpi=200, bbox_inches='tight')
            plt.close()

        # Rolling scatter (discrete coordinates)
        if all_roll:
            roll_out = pd.concat(all_roll, ignore_index=True)
            roll_out.to_csv(output_dir / 'similarity_rolling_metrics.csv', index=False)

            # Pretty scatter with colormap and trend bands

            # Choose metric to plot (default: ROC-AUC)
            ycol = scatter_metric.lower()
            if ycol not in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
                ycol = 'auc'
            label_map = {
                'auc': 'ROC-AUC',
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1': 'F1-score',
            }

            dfm = roll_out.dropna(subset=['x', ycol]).sort_values('x')
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
            sc = ax.scatter(dfm['x'], dfm[ycol], c=dfm['x'], cmap=cmap, s=point_size, alpha=alpha, edgecolors='none')
            if len(dfm) >= smooth_window:
                m = dfm[ycol].rolling(window=smooth_window, center=True, min_periods=max(3, smooth_window//3)).mean()
                s = dfm[ycol].rolling(window=smooth_window, center=True, min_periods=max(3, smooth_window//3)).std()
                ax.plot(dfm['x'], m, color='#1F4E79', linewidth=2)
                ax.fill_between(dfm['x'], (m - s).clip(lower=0.0), (m + s).clip(upper=1.0), color='#1F4E79', alpha=0.15)
            ax.set_title(f"{label_map[ycol]} vs NN Similarity (rolling)")
            ax.set_xlabel('ECFP4 Tanimoto (NN)')
            ax.set_ylabel(label_map[ycol])
            # Axis ranges (auto x based on data coverage if requested)
            if isinstance(xlim, str) and xlim.strip().lower() == 'auto':
                if not dfm['x'].empty:
                    xmin = float(dfm['x'].min())
                    xmax = float(dfm['x'].max())
                    pad = 0.02
                    xa = max(0.0, xmin - pad)
                    xb = min(1.0, xmax + pad)
                    ax.set_xlim(xa, xb)
            else:
                try:
                    xa, xb = [float(v) for v in xlim.split(',')]
                    ax.set_xlim(xa, xb)
                except Exception:
                    pass
            try:
                ya, yb = [float(v) for v in ylim_auc.split(',')]
                ax.set_ylim(ya, yb)
            except Exception:
                ax.set_ylim(0.0, 1.0)
            # Ticks format to look neat
            ax.set_yticks(np.linspace(0.5, 1.0, 6))
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
            cbar = fig.colorbar(sc, ax=ax, shrink=0.9, pad=0.02)
            cbar.set_label('NN Similarity')
            plt.tight_layout()
            plt.savefig(output_dir / 'similarity_scatter_metric_pretty.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Metric summary for reporting
            vals = dfm[ycol].to_numpy()
            metric_summary = {
                'metric': ycol,
                'n_points': int(vals.size),
                'mean': float(np.mean(vals)) if vals.size else None,
                'median': float(np.median(vals)) if vals.size else None,
                'p25': float(np.percentile(vals, 25)) if vals.size else None,
                'p75': float(np.percentile(vals, 75)) if vals.size else None,
                'below_0_5_count': int(np.sum(vals < 0.5)) if vals.size else 0,
            }
            (output_dir / 'similarity_metric_summary.json').write_text(json.dumps(metric_summary, indent=2))

            # Also render a pretty Accuracy scatter for comparison
            if 'accuracy' in roll_out.columns:
                acc_df = roll_out.dropna(subset=['x', 'accuracy']).sort_values('x')
                fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
                sca = ax.scatter(acc_df['x'], acc_df['accuracy'], c=acc_df['x'], cmap=cmap,
                                  s=point_size, alpha=alpha, edgecolors='none')
                if len(acc_df) >= smooth_window:
                    m = acc_df['accuracy'].rolling(window=smooth_window, center=True, min_periods=max(3, smooth_window//3)).mean()
                    s = acc_df['accuracy'].rolling(window=smooth_window, center=True, min_periods=max(3, smooth_window//3)).std()
                    ax.plot(acc_df['x'], m, color='#2D6A6D', linewidth=2)
                    ax.fill_between(acc_df['x'], (m - s).clip(lower=0.0), (m + s).clip(upper=1.0), color='#2D6A6D', alpha=0.15)
                ax.set_title('Accuracy vs NN Similarity (rolling)')
                ax.set_xlabel('ECFP4 Tanimoto (NN)')
                ax.set_ylabel('Accuracy')
                # X-range auto or manual
                if isinstance(xlim, str) and xlim.strip().lower() == 'auto':
                    if not acc_df['x'].empty:
                        xmin = float(acc_df['x'].min())
                        xmax = float(acc_df['x'].max())
                        pad = 0.02
                        xa = max(0.0, xmin - pad)
                        xb = min(1.0, xmax + pad)
                        ax.set_xlim(xa, xb)
                else:
                    try:
                        xa, xb = [float(v) for v in xlim.split(',')]
                        ax.set_xlim(xa, xb)
                    except Exception:
                        pass
                try:
                    ya, yb = [float(v) for v in ylim_acc.split(',')]
                    ax.set_ylim(ya, yb)
                except Exception:
                    ax.set_ylim(0.0, 1.0)
                cbar = fig.colorbar(sca, ax=ax, shrink=0.9, pad=0.02)
                cbar.set_label('NN Similarity')
                plt.tight_layout()
                plt.savefig(output_dir / 'similarity_scatter_accuracy_pretty.png', dpi=300, bbox_inches='tight')
                plt.close()

        # Pointwise overall plots per task (probability and correctness) if requested via CLI
        # Implemented in main() to avoid double IO here.
    except Exception as e:
        # Non-fatal if matplotlib is not available
        pass


def parse_bins(arg: str) -> List[float]:
    parts = [float(x) for x in arg.split(',') if x.strip()]
    if parts[0] > 0.0 or parts[-1] < 1.0:
        raise ValueError('Fixed bins must start at 0.0 and end at 1.0')
    return parts


def parse_percentiles(arg: str) -> List[float]:
    parts = [float(x) for x in arg.split(',') if x.strip()]
    if parts[0] < 0.0 or parts[-1] > 100.0:
        raise ValueError('Percentiles must be within [0, 100]')
    return parts


def main():
    parser = argparse.ArgumentParser(description='Analyze metrics across similarity bins (non-overlap subset)')
    parser.add_argument('--overlap_dir', type=str, default='tox21_overlap_check')
    parser.add_argument('--preds_dir', type=str, default='tox21_preds')
    parser.add_argument('--labels_dir', type=str, default='tox21 challenge')
    parser.add_argument('--output_dir', type=str, default='tox21_overlap_check/similarity_analysis')
    parser.add_argument('--fixed_bins', type=str, default='0.0,0.4,0.8,1.0')
    parser.add_argument('--percentiles', type=str, default='0,25,50,75,100')
    parser.add_argument('--rolling_window', type=int, default=0, help='Enable rolling metrics with given window size (e.g., 150)')
    parser.add_argument('--rolling_step', type=int, default=0, help='Stride between windows (default: window//5)')
    parser.add_argument('--style_nature', action='store_true', help='Use a clean, Nature-like style preset')
    parser.add_argument('--point_size', type=int, default=28, help='Scatter marker size')
    parser.add_argument('--alpha', type=float, default=0.6, help='Scatter alpha')
    parser.add_argument('--cmap', type=str, default='viridis', help='Matplotlib colormap name')
    parser.add_argument('--smooth_window', type=int, default=25, help='Smoothing window for trend band (points)')
    parser.add_argument('--scatter_metric', type=str, default='auc', choices=['auc','accuracy','precision','recall','f1'], help='Metric to plot on scatter (default: auc)')
    parser.add_argument('--xlim', type=str, default='auto', help='X-axis limits: "auto" or "min,max" (e.g., 0,1)')
    parser.add_argument('--ylim_auc', type=str, default='0.5,1.0', help='Y-axis limits for ROC-AUC scatter, e.g., "0.5,1.0"')
    parser.add_argument('--ylim_acc', type=str, default='0.5,1.0', help='Y-axis limits for Accuracy scatter, e.g., "0.5,1.0"')
    # Pointwise options
    parser.add_argument('--pointwise', action='store_true', help='Create pointwise probability and correctness scatter plots')
    parser.add_argument('--pointwise_tasks', type=str, default='nr-ahr', help='Comma-separated stems (e.g., "nr-ahr,sr-are") or "all"')
    parser.add_argument('--prob_threshold', type=float, default=0.5, help='Threshold for correctness (default 0.5)')
    parser.add_argument('--kernel_bw', type=float, default=0.08, help='Gaussian kernel bandwidth for smoothing')
    parser.add_argument('--jitter', type=float, default=0.04, help='Vertical jitter for correctness scatter')
    args = parser.parse_args()

    fixed_bins = parse_bins(args.fixed_bins)
    percentiles = parse_percentiles(args.percentiles)

    run_analysis(Path(args.overlap_dir), Path(args.preds_dir), Path(args.labels_dir), Path(args.output_dir), fixed_bins, percentiles,
                 args.rolling_window, args.rolling_step, args.style_nature, args.point_size, args.alpha, args.cmap, args.smooth_window,
                 args.scatter_metric, args.xlim, args.ylim_auc, args.ylim_acc)

    # Pointwise plots
    if args.pointwise:
        overlap_dir = Path(args.overlap_dir)
        preds_dir = Path(args.preds_dir)
        labels_dir = Path(args.labels_dir)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Allowed mapping with similarity
        allow_map = load_allowed_map(overlap_dir)
        stems = []
        if args.pointwise_tasks.strip().lower() == 'all':
            stems = list(STEM_TO_TASK.keys())
        else:
            stems = [s.strip().lower() for s in args.pointwise_tasks.split(',') if s.strip()]

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['font.sans-serif'] = ['Arial']
        mpl.rcParams['axes.unicode_minus'] = False
        if args.style_nature:
            mpl.rcParams.update({'figure.facecolor': 'white','axes.facecolor': 'white','axes.edgecolor': '#262626','axes.linewidth': 1.0})

        # Axis limits
        try:
            xa, xb = [float(v) for v in args.xlim.split(',')]
        except Exception:
            xa, xb = 0.0, 1.0

        for stem in stems:
            task = STEM_TO_TASK.get(stem)
            if task is None:
                continue
            pred_csv = preds_dir / f"{stem}_preds.csv"
            if not pred_csv.exists():
                matches = list(preds_dir.glob(f"*{stem}*.csv"))
                if not matches:
                    continue
                matches.sort(key=lambda p: len(p.name))
                pred_csv = matches[0]
            lab_path = labels_dir / f"{stem}.smiles"
            if not lab_path.exists():
                continue

            preds = pd.read_csv(pred_csv)
            labels = load_labels(lab_path)
            df = preds.merge(labels[['SMILES','label']], on='SMILES', how='inner')
            df = df.merge(allow_map[['smiles_raw','nn_sim']], left_on='SMILES', right_on='smiles_raw', how='inner')

            # Probability scatter
            if f'{task}_prob' in df.columns:
                fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))
                colors = np.where(df['label']>0, '#D62728', '#1F77B4')
                ax.scatter(df['nn_sim'], df[f'{task}_prob'], c=colors, s=args.point_size, alpha=args.alpha, edgecolors='none')
                # Smooth curves for positives and negatives
                xs = np.linspace(xa, xb, 200)
                for lbl, col in [(1,'#D62728'), (0,'#1F77B4')]:
                    sub = df[df['label']==lbl]
                    if not sub.empty:
                        sm = gaussian_smooth(sub['nn_sim'].to_numpy(), sub[f'{task}_prob'].to_numpy(), xs, bw=args.kernel_bw)
                        if np.any(~np.isnan(sm)):
                            ax.plot(xs, sm, color=col, linewidth=2.0, alpha=0.9, label=f'label={lbl}')
                ax.set_title(f'{task} Probability vs NN Similarity')
                ax.set_xlabel('ECFP4 Tanimoto (NN)')
                ax.set_ylabel('Predicted probability')
                ax.set_xlim(xa, xb)
                ax.set_ylim(0.0, 1.0)
                ax.legend(frameon=False)
                plt.tight_layout()
                plt.savefig(out / f'pointwise_prob_{stem}.png', dpi=300, bbox_inches='tight')
                plt.close()

            # Correctness scatter (thresholded)
            if task in df.columns or f'{task}_prob' in df.columns:
                proba = df.get(f'{task}_prob', None)
                if proba is not None:
                    pred_bin = (proba >= args.prob_threshold).astype(int)
                else:
                    pred_bin = df[task].astype(int)
                correct = (pred_bin.values == df['label'].astype(int).values).astype(float)
                # jitter
                jitter = (np.random.rand(correct.size) - 0.5) * 2 * args.jitter
                yj = correct + jitter

                fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))
                ax.scatter(df['nn_sim'], yj, c=df['label'].map({1:'#D62728',0:'#1F77B4'}), s=args.point_size, alpha=args.alpha, edgecolors='none')
                xs = np.linspace(xa, xb, 200)
                sm_all = gaussian_smooth(df['nn_sim'].to_numpy(), correct, xs, bw=args.kernel_bw)
                if np.any(~np.isnan(sm_all)):
                    ax.plot(xs, sm_all, color='#2D6A6D', linewidth=2.0, label='P(correct | sim)')
                ax.set_title(f'{task} Correctness vs NN Similarity')
                ax.set_xlabel('ECFP4 Tanimoto (NN)')
                ax.set_ylabel('Correct (0/1, jittered)')
                ax.set_xlim(xa, xb)
                ax.set_ylim(-0.1, 1.1)
                ax.legend(frameon=False)
                plt.tight_layout()
                plt.savefig(out / f'pointwise_correct_{stem}.png', dpi=300, bbox_inches='tight')
                plt.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
