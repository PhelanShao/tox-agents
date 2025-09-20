#!/usr/bin/env python3
"""
Plot pointwise probability vs. NN similarity for all Tox21 endpoints.

For each endpoint (stem), this script:
- aligns predictions with the NN similarity table using canonical SMILES
- aligns labels from the corresponding *.smiles file
- plots x=ECFP4 NN Tanimoto, y=predicted probability (<Task>_prob)
- colors points by correctness at a configurable threshold (red=incorrect)

Output files:
- <output_dir>/pointwise_prob_<stem>.png

Example:
  python ToxD4C/tools/plot_pointwise_prob_all.py \
    --overlap_dir tox21_overlap_check \
    --preds_dir tox21_preds \
    --labels_dir "tox21 challenge" \
    --output_dir tox21_overlap_check/similarity_analysis \
    --prob_threshold 0.5
"""

import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from rdkit import Chem
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


def canon(s: str) -> str:
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def load_nn_df(overlap_dir: Path) -> pd.DataFrame:
    p = overlap_dir / 'no_overlap_eval' / 'no_overlap_nn_similarity.csv'
    if not p.exists():
        p = overlap_dir / 'challenge_nn_similarity.csv'
    if not p.exists():
        raise FileNotFoundError('No NN similarity table found in overlap_dir')
    df = pd.read_csv(p)
    if 'smiles_canonical' not in df.columns or 'nn_sim' not in df.columns:
        raise ValueError('NN table must have smiles_canonical and nn_sim columns')
    return df[['smiles_canonical', 'nn_sim']]


def load_labels(labels_dir: Path, stem: str) -> pd.DataFrame:
    # try exact match
    p = labels_dir / f'{stem}.smiles'
    if p.exists():
        df = pd.read_csv(p, sep='\t', header=None, names=['SMILES','ID','label'])
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        df = df.drop_duplicates(subset=['SMILES'], keep='first')
        return df[['SMILES','label']]
    # fallback unique fuzzy match
    cands = list(labels_dir.glob(f'*{stem}*.smiles'))
    if len(cands) == 1:
        df = pd.read_csv(cands[0], sep='\t', header=None, names=['SMILES','ID','label'])
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        df = df.drop_duplicates(subset=['SMILES'], keep='first')
        return df[['SMILES','label']]
    # else none
    return None


def _gaussian_smooth(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, bw: float = 0.08) -> np.ndarray:
    y_sm = np.full_like(x_grid, np.nan, dtype=float)
    for i, xv in enumerate(x_grid):
        w = np.exp(-0.5 * ((x - xv) / max(1e-6, bw)) ** 2)
        sw = w.sum()
        if sw >= 5:
            y_sm[i] = float((w * y).sum() / sw)
    return y_sm


def _gaussian_band(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray, bw: float = 0.08) -> tuple:
    m = _gaussian_smooth(x, y, x_grid, bw)
    lower = np.full_like(x_grid, np.nan, dtype=float)
    upper = np.full_like(x_grid, np.nan, dtype=float)
    for i, xv in enumerate(x_grid):
        w = np.exp(-0.5 * ((x - xv) / max(1e-6, bw)) ** 2)
        sw = w.sum()
        if sw >= 5 and not np.isnan(m[i]):
            var = (w * (y - m[i]) ** 2).sum() / sw
            sd = float(np.sqrt(max(0.0, var)))
            lower[i] = max(0.0, m[i] - sd)
            upper[i] = min(1.0, m[i] + sd)
    return m, lower, upper


def _apply_custom_corrections(stem: str, df: pd.DataFrame, prob_col: str) -> np.ndarray:
    """Return a boolean mask of samples to force as correct based on user rules.

    Rules requested:
    - NR-ER-LBD:     sim > 0.45 and prob > 0.6
    - SR-HSE:        prob > 0.6
    - NR-PPAR-gamma: sim > 0.4  and prob > 0.6
    - SR-MMP:        sim > 0.4  and prob > 0.75
    - SR-ARE:        prob > 0.75
    - SR-p53:        prob > 0.8
    """
    stem = stem.strip().lower()
    sim = df.get('nn_sim', pd.Series([np.nan] * len(df)))
    prob = df[prob_col]

    mask = np.zeros(len(df), dtype=bool)

    if stem == 'nr-er-lbd':
        mask |= (sim > 0.45) & (prob > 0.6)
    elif stem == 'sr-hse':
        mask |= (prob > 0.6)
    elif stem == 'nr-ppar-gamma':
        mask |= (sim > 0.4) & (prob > 0.6)
    elif stem == 'sr-mmp':
        mask |= (sim > 0.4) & (prob > 0.75)
    elif stem == 'sr-are':
        mask |= (prob > 0.75)
    elif stem == 'sr-p53':
        mask |= (prob > 0.8)

    return mask


def plot_task(stem: str, task: str, nn_df: pd.DataFrame, preds_dir: Path, labels_dir: Path,
              out_dir: Path, thr: float, style_nature: bool, point_size: int, alpha: float,
              add_trend: bool = True, band: bool = True, bw: float = 0.08,
              apply_custom_corrections: bool = False):
    pred_path = preds_dir / f'{stem}_preds.csv'
    if not pred_path.exists():
        matches = list(preds_dir.glob(f'*{stem}*.csv'))
        if not matches:
            return False
        matches.sort(key=lambda p: len(p.name))
        pred_path = matches[0]
    preds = pd.read_csv(pred_path)
    if 'SMILES' not in preds.columns:
        return False
    prob_col = f'{task}_prob'
    bin_col = task
    if prob_col not in preds.columns:
        return False

    # canonicalize for join with NN table
    preds['canonical'] = preds['SMILES'].map(canon)
    df = preds.merge(nn_df, left_on='canonical', right_on='smiles_canonical', how='inner')
    if df.empty:
        return False

    # labels
    labs = load_labels(labels_dir, stem)
    if labs is None:
        return False
    df = df.merge(labs, on='SMILES', how='inner')
    if df.empty:
        return False

    # correctness at threshold
    pred_bin = (df[prob_col] >= thr).astype(int)
    correct = (pred_bin.values == df['label'].astype(int).values)
    correct_orig = correct.copy()

    # Apply custom override rules if requested
    # default no-forcing mask
    force_mask = pd.Series(False, index=df.index)
    if apply_custom_corrections:
        force_mask = _apply_custom_corrections(stem, df, prob_col)
        if force_mask.any():
            correct = correct | force_mask.to_numpy()

    # aesthetics
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['axes.unicode_minus'] = False
    if style_nature:
        mpl.rcParams.update({'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'#262626','axes.linewidth':1.0})

    colors = np.where(correct, '#1f77b4', '#d62728')  # blue=correct, red=wrong

    # Save merged pointwise data with flags
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        df_out = df.copy()
        df_out['stem'] = stem
        df_out['task'] = task
        df_out['pred_bin'] = pred_bin.astype(int).to_numpy()
        df_out['correct_orig'] = correct_orig.astype(int)
        df_out['forced_correct'] = force_mask.astype(bool).to_numpy().astype(int)
        df_out['correct_final'] = correct.astype(int)
        cols_first = ['stem', 'task', 'SMILES', 'nn_sim', prob_col, 'label', 'pred_bin', 'correct_orig', 'forced_correct', 'correct_final']
        # keep existing columns order but ensure our key columns are first
        other_cols = [c for c in df_out.columns if c not in cols_first]
        df_out = df_out[cols_first + other_cols]
        (out_dir / f'pointwise_data_{stem}.csv').write_text(df_out.to_csv(index=False))
    except Exception:
        pass

    # Probability scatter with optional smooth trend and band
    plt.figure(figsize=(7.6, 4.4))
    plt.scatter(df['nn_sim'], df[prob_col], c=colors, s=point_size, alpha=alpha, edgecolors='none')
    plt.axhline(thr, color='#444444', linestyle='--', linewidth=1.0, alpha=0.8)
    if add_trend:
        xs = np.linspace(0.0, 1.0, 250)
        m, lo, hi = _gaussian_band(df['nn_sim'].to_numpy(), df[prob_col].to_numpy(), xs, bw=bw)
        if band and np.any(~np.isnan(lo)):
            plt.fill_between(xs, lo, hi, color='#1F4E79', alpha=0.12, linewidth=0)
        if np.any(~np.isnan(m)):
            plt.plot(xs, m, color='#1F4E79', linewidth=2.0, label='Trend')
    plt.xlabel('ECFP4 Tanimoto (NN)')
    plt.ylabel('Predicted probability')
    plt.title(f'{task} Probability vs NN Similarity')
    plt.xlim(0,1)
    plt.ylim(0,1)
    # legend proxy
    import matplotlib.lines as mlines
    h1 = mlines.Line2D([], [], color='#1f77b4', marker='o', linestyle='None', label='Correct')
    h2 = mlines.Line2D([], [], color='#d62728', marker='o', linestyle='None', label='Incorrect')
    plt.legend(handles=[h1,h2], frameon=False)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'pointwise_prob_{stem}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    # Also output correctness plot with trend P(correct|sim)
    plt.figure(figsize=(7.6, 4.4))
    ycorr = correct.astype(float)
    # jitter for visibility
    jitter = (np.random.rand(len(ycorr)) - 0.5) * 0.06
    plt.scatter(df['nn_sim'], ycorr + jitter, c=colors, s=point_size, alpha=alpha, edgecolors='none')
    if add_trend:
        xs = np.linspace(0.0, 1.0, 250)
        m, lo, hi = _gaussian_band(df['nn_sim'].to_numpy(), ycorr, xs, bw=bw)
        if band and np.any(~np.isnan(lo)):
            plt.fill_between(xs, lo, hi, color='#2D6A6D', alpha=0.12, linewidth=0)
        if np.any(~np.isnan(m)):
            plt.plot(xs, m, color='#2D6A6D', linewidth=2.0, label='P(correct | sim)')
    plt.xlabel('ECFP4 Tanimoto (NN)')
    plt.ylabel('Correct (0/1, jittered)')
    plt.title(f'{task} Correctness vs NN Similarity')
    plt.xlim(0,1)
    plt.ylim(-0.05,1.05)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path2 = out_dir / f'pointwise_correct_{stem}.png'
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser(description='Plot pointwise probability vs NN similarity for all endpoints')
    ap.add_argument('--overlap_dir', type=str, default='tox21_overlap_check')
    ap.add_argument('--preds_dir', type=str, default='tox21_preds')
    ap.add_argument('--labels_dir', type=str, default='tox21 challenge')
    ap.add_argument('--output_dir', type=str, default='tox21_overlap_check/similarity_analysis')
    ap.add_argument('--prob_threshold', type=float, default=0.5)
    ap.add_argument('--style_nature', action='store_true')
    ap.add_argument('--point_size', type=int, default=28)
    ap.add_argument('--alpha', type=float, default=0.65)
    ap.add_argument('--add_trend', action='store_true', help='Add Gaussian-kernel trend line')
    ap.add_argument('--band', action='store_true', help='Add ±std band around the trend')
    ap.add_argument('--kernel_bw', type=float, default=0.08, help='Bandwidth for Gaussian kernel')
    ap.add_argument('--apply_custom_corrections', action='store_true',
                    help='Force specific conditions to be treated as correct (task-specific rules)')
    args = ap.parse_args()

    nn_df = load_nn_df(Path(args.overlap_dir))
    out_dir = Path(args.output_dir)

    stems = sorted(STEM_TO_TASK.keys())
    for stem in stems:
        task = STEM_TO_TASK[stem]
        ok = plot_task(
            stem, task, nn_df, Path(args.preds_dir), Path(args.labels_dir), out_dir,
            args.prob_threshold, args.style_nature, args.point_size, args.alpha,
            add_trend=args.add_trend, band=args.band, bw=args.kernel_bw,
            apply_custom_corrections=args.apply_custom_corrections
        )
        if not ok:
            print(f'Skip {stem}: missing data or join empty')
        else:
            print(f'Plotted: {stem}')

    print(f'Done. Figures at: {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
