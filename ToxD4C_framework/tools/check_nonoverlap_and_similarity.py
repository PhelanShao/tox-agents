#!/usr/bin/env python3
"""
Check non-overlap between training LMDB and an external challenge set (e.g., tox21),
using canonical SMILES → standard InChIKey mapping and scaffold exclusion;
optionally compute nearest-neighbor ECFP4 Tanimoto similarities.

Outputs summary CSVs for identifiers, scaffolds, and similarity stats.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs

# Project imports
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from data.lmdb_dataset import LMDBToxD4CDataset


def parse_smiles_lines(lines: List[str]) -> List[str]:
    smiles = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        if 'smiles' in low and (low.startswith('smiles') or low.split()[0] == 'smiles'):
            continue
        parts = s.split('\t') if ('\t' in s) else s.split()
        if parts:
            smiles.append(parts[0])
    return smiles


def standardize_mol(smi: str) -> Optional[Chem.Mol]:
    """Parse SMILES, keep largest fragment, sanitize.

    Uses RDKit's GetMolFrags with correct keyword 'sanitizeFrags' to support
    different RDKit versions.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # Keep the largest fragment to remove salts
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        except TypeError:
            # Fallback to positional args: (mol, asMols, sanitizeFrags)
            frags = Chem.GetMolFrags(mol, True, True)
        if len(frags) > 1:
            frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
            mol = frags[0]
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_to_canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True)


def mol_to_inchikey(mol: Chem.Mol) -> Optional[str]:
    try:
        inchi = Chem.MolToInchi(mol)  # standard InChI by default
        ikey = Chem.InchiToInchiKey(inchi)
        return ikey
    except Exception:
        return None


def mol_to_scaffolds(mol: Chem.Mol) -> Tuple[Optional[str], Optional[str]]:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaf_smiles = Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        scaf_smiles = None
    try:
        gen = MurckoScaffold.MakeScaffoldGeneric(scaf) if scaf_smiles is not None else None
        gen_smiles = Chem.MolToSmiles(gen, canonical=True) if gen is not None else None
    except Exception:
        gen_smiles = None
    return scaf_smiles, gen_smiles


def mol_to_ecfp4(mol: Chem.Mol, n_bits: int = 2048, use_chirality: bool = False):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits, useChirality=use_chirality)


def collect_train_smiles(lmdb_dir: Path) -> List[str]:
    train_path = lmdb_dir / 'train.lmdb'
    ds = LMDBToxD4CDataset(str(train_path))
    return list(ds.smiles_keys)


def load_challenge_smiles(challenge_dir: Path) -> List[str]:
    all_smiles: List[str] = []
    for fp in sorted(challenge_dir.glob('*.smiles')):
        lines = fp.read_text(encoding='utf-8', errors='ignore').splitlines()
        smi = parse_smiles_lines(lines)
        all_smiles.extend(smi)
    # Deduplicate early on SMILES text
    return list(dict.fromkeys(all_smiles))


def build_index(smiles_list: List[str]) -> pd.DataFrame:
    rows = []
    for s in smiles_list:
        mol = standardize_mol(s)
        if mol is None:
            continue
        cano = mol_to_canonical_smiles(mol)
        ikey = mol_to_inchikey(mol)
        scaf, gen_scaf = mol_to_scaffolds(mol)
        rows.append({
            'smiles_raw': s,
            'smiles_canonical': cano,
            'inchikey': ikey,
            'scaffold': scaf,
            'scaffold_generic': gen_scaf,
        })
    return pd.DataFrame(rows)


def compute_nn_similarity(train_df: pd.DataFrame, chall_df: pd.DataFrame, n_bits: int = 2048, use_chirality: bool = False) -> pd.DataFrame:
    # Precompute mol + fp for train
    train_fps = []
    train_keys = []
    for s in train_df['smiles_canonical']:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            train_fps.append(None)
            train_keys.append(None)
            continue
        fp = mol_to_ecfp4(mol, n_bits=n_bits, use_chirality=use_chirality)
        train_fps.append(fp)
        train_keys.append(s)
    # Filter None
    tfps = [fp for fp in train_fps if fp is not None]
    tkeys = [k for fp, k in zip(train_fps, train_keys) if fp is not None]

    out_rows = []
    for s, ikey in zip(chall_df['smiles_canonical'], chall_df['inchikey']):
        mol = Chem.MolFromSmiles(s)
        if mol is None or not tfps:
            out_rows.append({'smiles_canonical': s, 'inchikey': ikey, 'nn_sim': np.nan, 'nn_train_smiles': None})
            continue
        qfp = mol_to_ecfp4(mol, n_bits=n_bits, use_chirality=use_chirality)
        sims = DataStructs.BulkTanimotoSimilarity(qfp, tfps)
        if not sims:
            out_rows.append({'smiles_canonical': s, 'inchikey': ikey, 'nn_sim': np.nan, 'nn_train_smiles': None})
            continue
        best_idx = int(np.argmax(sims))
        out_rows.append({'smiles_canonical': s, 'inchikey': ikey, 'nn_sim': float(sims[best_idx]), 'nn_train_smiles': tkeys[best_idx]})
    return pd.DataFrame(out_rows)


def main():
    parser = argparse.ArgumentParser(description='Non-overlap check and ECFP4 similarity')
    parser.add_argument('--lmdb_dir', type=str, default='data/data/processed', help='Directory containing train.lmdb')
    parser.add_argument('--challenge_dir', type=str, default='tox21 challenge', help='Directory with tox21 *.smiles')
    parser.add_argument('--output_dir', type=str, default='tox21_overlap_check', help='Directory to save outputs')
    parser.add_argument('--compute_similarity', action='store_true', help='Compute ECFP4 nearest-neighbor similarities')
    parser.add_argument('--use_chirality', action='store_true', help='Use chiral ECFP4 for similarity (useChirality=True)')
    args = parser.parse_args()

    lmdb_dir = Path(args.lmdb_dir)
    challenge_dir = Path(args.challenge_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SMILES
    print('Loading training SMILES from LMDB...')
    train_smiles = collect_train_smiles(lmdb_dir)
    print(f'  Train molecules: {len(train_smiles)}')

    print('Loading challenge SMILES...')
    chall_smiles = load_challenge_smiles(challenge_dir)
    print(f'  Challenge molecules (unique text): {len(chall_smiles)}')

    # Build indices
    print('Standardizing and building indices...')
    train_df = build_index(train_smiles)
    chall_df = build_index(chall_smiles)

    # Save indices
    train_df.to_csv(out_dir / 'train_index.csv', index=False)
    chall_df.to_csv(out_dir / 'challenge_index.csv', index=False)

    # Direct identity check by InChIKey plus canonical smiles (union)
    train_ids = set(train_df['inchikey'].dropna().tolist())
    chall_ids = set(chall_df['inchikey'].dropna().tolist())
    overlap_ids = train_ids.intersection(chall_ids)

    train_canos = set(train_df['smiles_canonical'].dropna().tolist())
    chall_canos = set(chall_df['smiles_canonical'].dropna().tolist())
    overlap_canos = train_canos.intersection(chall_canos)

    direct_overlap = sorted(overlap_ids.union(overlap_canos))

    # Scaffold exclusion (generic Bemis–Murcko)
    train_scaf = set([s for s in train_df['scaffold_generic'].dropna().tolist()])
    chall_scaf = set([s for s in chall_df['scaffold_generic'].dropna().tolist()])
    scaf_overlap = sorted(train_scaf.intersection(chall_scaf))

    # Summaries
    summary = {
        'train_n': int(len(train_df)),
        'challenge_n': int(len(chall_df)),
        'direct_overlap_count': int(len(direct_overlap)),
        'scaffold_overlap_count': int(len(scaf_overlap)),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('Summary:')
    print(json.dumps(summary, indent=2))

    # Save overlap lists (for audit)
    (out_dir / 'direct_overlap.txt').write_text('\n'.join(direct_overlap))
    (out_dir / 'scaffold_overlap.txt').write_text('\n'.join(scaf_overlap))

    # Similarity (optional)
    if args.compute_similarity:
        print('Computing ECFP4 nearest-neighbor Tanimoto...')
        nn_df = compute_nn_similarity(train_df, chall_df, n_bits=2048, use_chirality=args.use_chirality)
        nn_df.to_csv(out_dir / 'challenge_nn_similarity.csv', index=False)

        # Distribution summary
        sims = nn_df['nn_sim'].dropna().to_numpy()
        dist = {
            'n': int(sims.size),
            'mean': float(np.mean(sims)) if sims.size else None,
            'median': float(np.median(sims)) if sims.size else None,
            'p95': float(np.percentile(sims, 95)) if sims.size else None,
            'p99': float(np.percentile(sims, 99)) if sims.size else None,
            'max': float(np.max(sims)) if sims.size else None,
        }
        (out_dir / 'similarity_stats.json').write_text(json.dumps(dist, indent=2))
        print('Similarity stats:')
        print(json.dumps(dist, indent=2))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
