#!/usr/bin/env python3
"""
Export, for each Tox21 endpoint, the Top-K most structurally dissimilar
challenge molecules (to the training set) as XYZ files.

Selection is based on the precomputed nearest-neighbor table produced by
tools/check_nonoverlap_and_similarity.py: the lower the NN ECFP4 Tanimoto,
the more dissimilar.

Usage example:
  python ToxD4C/tools/export_topk_xyz_per_task.py \
    --overlap_dir tox21_overlap_check \
    --labels_dir "tox21 challenge" \
    --output_dir tox21_overlap_check/topk_xyz \
    --top_k 5

Outputs: one folder per endpoint (stem of .smiles file) containing K xyz files
and a small CSV index with SMILES and similarity values.
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


def parse_smiles_lines(path: Path) -> List[str]:
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    out = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        if 'smiles' in low and (low.startswith('smiles') or low.split()[0] == 'smiles'):
            continue
        parts = s.split('\t') if ('\t' in s) else s.split()
        if parts:
            out.append(parts[0])
    # dedup keep order
    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def to_canonical(smiles: List[str]) -> List[str]:
    cano = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        cano.append(Chem.MolToSmiles(m, canonical=True))
    return cano


def make_xyz(mol: Chem.Mol, comment: str = "") -> str:
    # Ensure 3D
    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(m, params) == -1:
        # fallback random
        params.useRandomCoords = True
        AllChem.EmbedMolecule(m, params)
    try:
        AllChem.UFFOptimizeMolecule(m, maxIters=200)
    except Exception:
        pass
    conf = m.GetConformer()
    n = m.GetNumAtoms()
    lines = [str(n), comment]
    for i in range(n):
        a = m.GetAtomWithIdx(i)
        sym = a.GetSymbol()
        pos = conf.GetAtomPosition(i)
        lines.append(f"{sym} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines) + "\n"


def load_nn_table(overlap_dir: Path) -> pd.DataFrame:
    # Prefer filtered non-overlap table if available
    p = overlap_dir / 'no_overlap_eval' / 'no_overlap_nn_similarity.csv'
    if not p.exists():
        p = overlap_dir / 'challenge_nn_similarity.csv'
    if not p.exists():
        raise FileNotFoundError('Nearest-neighbor table not found under overlap_dir')
    df = pd.read_csv(p)
    if 'smiles_canonical' not in df.columns or 'nn_sim' not in df.columns:
        raise ValueError('NN table missing required columns: smiles_canonical, nn_sim')
    return df[['smiles_canonical', 'inchikey' if 'inchikey' in df.columns else df.columns[1], 'nn_sim', 'nn_train_smiles' if 'nn_train_smiles' in df.columns else None]].rename(columns=lambda x: 'inchikey' if x and 'inchikey' in x else x)


def export_task_topk(stem: str, smiles_path: Path, nn_df: pd.DataFrame, out_dir: Path, top_k: int) -> Tuple[int, Path]:
    smiles = parse_smiles_lines(smiles_path)
    cano = to_canonical(smiles)
    sub = nn_df[nn_df['smiles_canonical'].isin(cano)].copy()
    if sub.empty:
        return 0, out_dir
    sub = sub.sort_values('nn_sim', ascending=True).head(top_k)

    task_dir = out_dir / stem
    task_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for idx, row in sub.iterrows():
        smi = row['smiles_canonical']
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        xyz = make_xyz(m, comment=f"{stem} sim={row['nn_sim']:.4f}")
        fn = task_dir / f"rank{len(records)+1:02d}_sim{row['nn_sim']:.3f}.xyz"
        fn.write_text(xyz)
        records.append({'rank': len(records)+1, 'smiles_canonical': smi, 'nn_sim': float(row['nn_sim']), 'xyz': str(fn)})

    if records:
        pd.DataFrame(records).to_csv(task_dir / 'index.csv', index=False)
    return len(records), task_dir


def main():
    parser = argparse.ArgumentParser(description='Export top-K structurally most (dis)similar molecules per endpoint as XYZ')
    parser.add_argument('--overlap_dir', type=str, default='tox21_overlap_check')
    parser.add_argument('--labels_dir', type=str, default='tox21 challenge')
    parser.add_argument('--output_dir', type=str, default='tox21_overlap_check/topk_xyz')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--tasks', type=str, default='all', help='Comma-separated stems (e.g., "nr-ahr,sr-are") or "all"')
    parser.add_argument('--mode', type=str, default='low', choices=['low','high'], help='Pick lowest (dissimilar) or highest (similar) NN similarity')
    parser.add_argument('--with_train_xyz', action='store_true', help='Also export the matched training NN molecules as XYZ')
    args = parser.parse_args()

    overlap_dir = Path(args.overlap_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nn_df = load_nn_table(overlap_dir)

    stems = []
    if args.tasks.strip().lower() == 'all':
        stems = [p.stem for p in labels_dir.glob('*.smiles')]
    else:
        stems = [s.strip() for s in args.tasks.split(',') if s.strip()]

    summary = []
    for stem in sorted(stems):
        smiles_path = labels_dir / f"{stem}.smiles"
        if not smiles_path.exists():
            # try alternative naming if needed
            candidates = list(labels_dir.glob(f"*{stem}*.smiles"))
            if not candidates:
                continue
            smiles_path = candidates[0]
        # select direction
        sel = nn_df[nn_df['smiles_canonical'].isin(to_canonical(parse_smiles_lines(smiles_path)))].copy()
        if sel.empty:
            summary.append({'task': stem, 'exported': 0, 'folder': str(out_dir / stem)})
            continue
        ascending = True if args.mode == 'low' else False
        sel = sel.sort_values('nn_sim', ascending=ascending).head(args.top_k)

        task_dir = out_dir / stem
        task_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for rank, row in enumerate(sel.itertuples(index=False), start=1):
            smi = row.smiles_canonical
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            xyz = make_xyz(m, comment=f"{stem} sim={row.nn_sim:.4f}")
            fn = task_dir / f"rank{rank:02d}_sim{row.nn_sim:.3f}.xyz"
            fn.write_text(xyz)
            rec = {'rank': rank, 'smiles_canonical': smi, 'nn_sim': float(row.nn_sim), 'xyz': str(fn)}
            # export train counterpart if requested and available
            if args.with_train_xyz and hasattr(row, 'nn_train_smiles') and isinstance(row.nn_train_smiles, str):
                tm = Chem.MolFromSmiles(row.nn_train_smiles)
                if tm is not None:
                    txyz = make_xyz(tm, comment=f"trainNN sim={row.nn_sim:.4f}")
                    tfn = task_dir / f"rank{rank:02d}_trainNN_sim{row.nn_sim:.3f}.xyz"
                    tfn.write_text(txyz)
                    rec['train_xyz'] = str(tfn)
                    rec['train_smiles'] = row.nn_train_smiles
            records.append(rec)

        if records:
            pd.DataFrame(records).to_csv(task_dir / 'index.csv', index=False)
        summary.append({'task': stem, 'exported': len(records), 'folder': str(task_dir)})

    pd.DataFrame(summary).to_csv(out_dir / 'summary.csv', index=False)
    print(f"Done. Summary saved to: {out_dir / 'summary.csv'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
