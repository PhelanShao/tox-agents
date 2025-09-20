#!/usr/bin/env python3
"""
Visualize structurally most dissimilar molecules (to the training set) via ECFP4
similarity maps. Uses the precomputed nearest-neighbor table produced by
tools/check_nonoverlap_and_similarity.py.

Features
- Select top-K outliers by lowest NN similarity (ECFP4 Tanimoto)
- For each outlier, draw an atom-wise similarity map against its NN train molecule
- Optionally, do this per Tox21 task (based on prediction files) or globally
- Save individual PNGs and per-task grid collages

Usage example
  python ToxD4C/tools/visualize_structural_outliers.py \
    --overlap_dir tox21_overlap_check \
    --output_dir tox21_overlap_check/outliers_viz \
    --top_k 16 --per_task

Requirements: RDKit (SimilarityMaps), pandas, numpy, matplotlib.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
import matplotlib as mpl
from rdkit.Chem import AllChem
from rdkit import DataStructs
import matplotlib.pyplot as plt


def load_nn_table(overlap_dir: Path) -> pd.DataFrame:
    nn_path = overlap_dir / 'challenge_nn_similarity.csv'
    if not nn_path.exists():
        # fallback to filtered table, if available
        nn_path = overlap_dir / 'no_overlap_eval' / 'no_overlap_nn_similarity.csv'
    if not nn_path.exists():
        raise FileNotFoundError('Nearest-neighbor table not found. Run check_nonoverlap_and_similarity.py first.')
    df = pd.read_csv(nn_path)
    # expected columns: smiles_canonical, inchikey, nn_sim, nn_train_smiles
    if 'nn_train_smiles' not in df.columns:
        raise ValueError('nn_train_smiles column missing; re-run similarity with a recent script version.')
    return df


def get_mol(smi: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    AllChem.Compute2DCoords(m)
    return m


def draw_similarity_map(probe: Chem.Mol, ref: Chem.Mol, out_png: Path,
                        radius: int = 2, n_bits: int = 2048, use_chirality: bool = False,
                        cmap = plt.cm.viridis, dpi: int = 300, width: int = 900, height: int = 900,
                        nn_sim_value: float = 1.0, alpha_base: float = 0.6, contour_lines: int = 12):
    """Draw atom-wise SimilarityMap using ECFP4 and Tanimoto against a reference molecule."""
    if probe is None or ref is None:
        return
    # Build fp function wrapper that accepts (mol, atomId)
    fp_func = lambda m, a=None: SimilarityMaps.GetMorganFingerprint(
        m, atomId=a if a is not None else -1, radius=radius, fpType='bv', useChirality=use_chirality, nBits=n_bits
    )

    # Some RDKit builds require an explicit RDKit drawer (no matplotlib figure).
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    # Ensure a white background (not the low-value color of the map)
    try:
        opts = drawer.drawOptions()
        # White background
        if hasattr(opts, 'setBackgroundColour') and callable(getattr(opts, 'setBackgroundColour')):
            opts.setBackgroundColour((1.0, 1.0, 1.0))
        else:
            opts.backgroundColour = (1.0, 1.0, 1.0)
        opts.clearBackground = True
        # Add padding to prevent truncation of contours
        if hasattr(opts, 'padding'):
            opts.padding = 0.20  # 20% border
    except Exception:
        pass

    # RDKit expects colorMap as a 3-tuple of colors (low, mid, high).
    # If a Matplotlib colormap is provided, sample it at 0.0/0.5/1.0.
    if isinstance(cmap, mpl.colors.Colormap):
        # Force low & mid to white so near-zero regions remain white
        c0 = (1.0, 1.0, 1.0)
        c1 = (1.0, 1.0, 1.0)
        # warm highlight for high values
        c2 = (1.0, 0.85, 0.2)
        rdkit_cmap = (c0, c1, c2)
    elif isinstance(cmap, (list, tuple)) and len(cmap) >= 3:
        rdkit_cmap = tuple(cmap[:3])
    else:
        # fallback to a pleasant default
        rdkit_cmap = ((0.267, 0.004, 0.329), (0.127, 0.566, 0.550), (0.993, 0.906, 0.144))
    # Dynamic alpha: lower intensity for dissimilar pairs
    dyn_alpha = max(0.10, min(1.0, alpha_base * float(nn_sim_value)))
    SimilarityMaps.GetSimilarityMapForFingerprint(
        probe, ref,
        fpFunction=fp_func,
        metric=DataStructs.TanimotoSimilarity,
        colorMap=rdkit_cmap,
        draw2d=drawer,
        alpha=dyn_alpha,
        contourLines=contour_lines
    )
    drawer.FinishDrawing()
    out_png.write_bytes(drawer.GetDrawingText())


def tile_images(img_paths: List[Path], out_png: Path, cols: int = 4, title: str = None):
    imgs = [plt.imread(str(p)) for p in img_paths if p.exists()]
    if not imgs:
        return
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.0, rows*3.0))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)
    for ax, img in zip(axes.ravel(), imgs):
        ax.imshow(img)
        ax.axis('off')
    for ax in axes.ravel()[len(imgs):]:
        ax.axis('off')
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(str(out_png), dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize structural outliers via ECFP similarity maps')
    parser.add_argument('--overlap_dir', type=str, default='tox21_overlap_check', help='Directory with NN similarity table')
    parser.add_argument('--output_dir', type=str, default='tox21_overlap_check/outliers_viz', help='Output directory')
    parser.add_argument('--top_k', type=int, default=16, help='Top-K molecules to visualize')
    parser.add_argument('--mode', type=str, default='low', choices=['low','high'], help='Select most dissimilar (low) or most similar (high) by NN similarity')
    parser.add_argument('--per_task', action='store_true', help='Select outliers per tox21 task based on predictions present')
    parser.add_argument('--preds_dir', type=str, default='tox21_preds', help='Predictions dir to define per-task sets')
    parser.add_argument('--use_chirality', action='store_true', help='Use chiral ECFP4 for similarity coloring')
    parser.add_argument('--img_width', type=int, default=1200, help='Output image width (px)')
    parser.add_argument('--img_height', type=int, default=900, help='Output image height (px)')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI (only for grid image)')
    parser.add_argument('--alpha_base', type=float, default=0.6, help='Base alpha, scaled by NN similarity')
    parser.add_argument('--contours', type=int, default=12, help='Number of contour lines')
    args = parser.parse_args()

    overlap_dir = Path(args.overlap_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nn_df = load_nn_table(overlap_dir)

    ascending = True if args.mode == 'low' else False

    if args.per_task:
        preds_dir = Path(args.preds_dir)
        # determine stems from available preds files
        stems = [p.stem.replace('_preds','') for p in preds_dir.glob('*_preds.csv')]
        for stem in stems:
            # map stem to subset by joining with corresponding label file if needed; here we just use preds SMILES intersection
            pred_path = preds_dir / f"{stem}_preds.csv"
            if not pred_path.exists():
                continue
            pred = pd.read_csv(pred_path)[['SMILES']]
            sub = nn_df.merge(pred, left_on='smiles_canonical', right_on='SMILES', how='inner')
            sub = sub.sort_values('nn_sim', ascending=ascending).head(args.top_k)
            if sub.empty:
                continue
            task_out = out_dir / stem / args.mode
            task_out.mkdir(parents=True, exist_ok=True)
            img_paths = []
            for i, row in sub.iterrows():
                probe = get_mol(row['smiles_canonical'])
                ref = get_mol(row['nn_train_smiles'])
                out_png = task_out / f"{args.mode}_{i:04d}_sim_{row['nn_sim']:.3f}.png"
                # verify sim between the exact pair
                try:
                    from rdkit.Chem import rdMolDescriptors as rdd
                    fp1 = rdd.GetMorganFingerprintAsBitVect(probe, 2, nBits=2048, useChirality=args.use_chirality)
                    fp2 = rdd.GetMorganFingerprintAsBitVect(ref, 2, nBits=2048, useChirality=args.use_chirality)
                    sim_check = DataStructs.TanimotoSimilarity(fp1, fp2)
                except Exception:
                    sim_check = float(row['nn_sim'])
                draw_similarity_map(
                    probe, ref, out_png,
                    use_chirality=args.use_chirality,
                    width=args.img_width, height=args.img_height, dpi=args.dpi,
                    nn_sim_value=sim_check, alpha_base=args.alpha_base, contour_lines=args.contours
                )
                img_paths.append(out_png)
            title = f"{stem}: {'most dissimilar' if args.mode=='low' else 'most similar'} top-{args.top_k}"
            tile_images(img_paths, task_out / f"{stem}_{args.mode}_top{args.top_k}_grid.png", cols=4, title=title)
    else:
        # global selection across all challenge molecules
        sub = nn_df.sort_values('nn_sim', ascending=ascending).head(args.top_k)
        img_paths = []
        for i, row in sub.iterrows():
            probe = get_mol(row['smiles_canonical'])
            ref = get_mol(row['nn_train_smiles'])
            out_png = out_dir / f"{args.mode}_{i:04d}_{row['inchikey'] if 'inchikey' in row and pd.notna(row['inchikey']) else 'mol'}_sim_{row['nn_sim']:.3f}.png"
            draw_similarity_map(
                probe, ref, out_png,
                use_chirality=args.use_chirality,
                width=args.img_width, height=args.img_height, dpi=args.dpi,
                nn_sim_value=row['nn_sim'], alpha_base=args.alpha_base, contour_lines=args.contours
            )
            img_paths.append(out_png)
        title = f"Global {'most dissimilar' if args.mode=='low' else 'most similar'} top-{args.top_k}"
        tile_images(img_paths, out_dir / f"global_{args.mode}_top{args.top_k}_grid.png", cols=4, title=title)

    print(f"Saved visualizations to: {out_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
