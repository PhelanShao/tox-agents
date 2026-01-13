"""
Utilities for dataset splitting strategies and LMDB materialization.

Implements Bemis–Murcko scaffold-based splitting for molecular datasets and
helpers to build new LMDB splits from existing LMDBs.

This module is used by train.py to support --split_method scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import lmdb
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def bemis_murcko_scaffold(smiles: str) -> Optional[str]:
    """Compute the Bemis–Murcko scaffold SMILES for a given molecule.

    Returns None if the SMILES cannot be parsed or scaffold cannot be produced.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return None
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


@dataclass
class SplitIndices:
    train: List[int]
    valid: List[int]
    test: List[int]

    def to_json(self) -> Dict[str, List[int]]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


def scaffold_split_indices(
    smiles_list: Sequence[str],
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> SplitIndices:
    """Create non-overlapping scaffold-based splits for the given SMILES list.

    - Groups molecules by Bemis–Murcko scaffold.
    - Sorts scaffold groups by size (desc) to place large scaffolds first.
    - Greedily assigns each scaffold group to the split with the largest remaining capacity.
    """
    assert abs(sum(frac) - 1.0) < 1e-6, "fractions must sum to 1.0"
    n = len(smiles_list)
    rng = np.random.RandomState(seed)

    # Build scaffold groups
    scaf_to_indices: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles_list):
        scaf = bemis_murcko_scaffold(smi)
        key = scaf if scaf is not None else f"_NOSCAF_{i}"
        scaf_to_indices.setdefault(key, []).append(i)

    # Shuffle groups with a deterministic seed for tie-breaking
    groups = list(scaf_to_indices.items())
    rng.shuffle(groups)
    # Sort by group size descending
    groups.sort(key=lambda kv: len(kv[1]), reverse=True)

    # Target sizes
    n_train = int(round(frac[0] * n))
    n_valid = int(round(frac[1] * n))
    n_test = n - n_train - n_valid

    train_idx: List[int] = []
    valid_idx: List[int] = []
    test_idx: List[int] = []

    def remaining() -> Tuple[int, int, int]:
        return (n_train - len(train_idx), n_valid - len(valid_idx), n_test - len(test_idx))

    for _, idxs in groups:
        r_train, r_valid, r_test = remaining()
        # Choose split with max remaining capacity
        rem = np.array([r_train, r_valid, r_test])
        target_split = int(rem.argmax())
        if target_split == 0:
            train_idx.extend(idxs)
        elif target_split == 1:
            valid_idx.extend(idxs)
        else:
            test_idx.extend(idxs)

    return SplitIndices(train=sorted(train_idx), valid=sorted(valid_idx), test=sorted(test_idx))


def verify_no_scaffold_overlap(smiles_list: Sequence[str], splits: SplitIndices) -> bool:
    """Verify that no scaffold appears in more than one split."""
    split_map = {
        "train": set(splits.train),
        "valid": set(splits.valid),
        "test": set(splits.test),
    }
    index_to_scaf: Dict[int, Optional[str]] = {
        i: bemis_murcko_scaffold(smi) for i, smi in enumerate(smiles_list)
    }
    scaf_to_split: Dict[Optional[str], str] = {}
    for split_name, idxs in split_map.items():
        for i in idxs:
            scaf = index_to_scaf[i]
            if scaf in scaf_to_split and scaf_to_split[scaf] != split_name:
                return False
            scaf_to_split[scaf] = split_name
    return True


def _collect_all_entries_from_raw_lmdb(raw_dir: Path) -> Dict[str, bytes]:
    """Read and merge entries from LMDB splits under raw_dir.

    Returns a dict mapping SMILES -> pickled sample bytes from the first seen split.
    """
    raw_dir = Path(raw_dir)
    result: Dict[str, bytes] = {}
    for split in ("train.lmdb", "valid.lmdb", "test.lmdb"):
        env_path = raw_dir / split
        if not env_path.exists():
            continue
        env = lmdb.open(str(env_path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    key_str = key.decode("ascii")
                except Exception:
                    continue
                if not key_str.isdigit() and key_str != "length":
                    result.setdefault(key_str, value)
        env.close()
    return result


def _write_lmdb_from_entries(output_path: Path, entries: Dict[str, bytes]) -> None:
    """Write a new LMDB at output_path using the provided SMILES->bytes entries."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Generous map size; adjust if needed
    env = lmdb.open(str(output_path), map_size=50 * 1024 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for smi, data in entries.items():
            txn.put(smi.encode("ascii", errors="ignore"), data)
    env.sync()
    env.close()


def _collect_all_entries_from_processed_lmdb_dir(lmdb_dir: Path) -> Dict[str, bytes]:
    """Collect SMILES->bytes from {train,valid,test}.lmdb (directory form, subdir=True)."""
    lmdb_dir = Path(lmdb_dir)
    result: Dict[str, bytes] = {}
    for split in ("train.lmdb", "valid.lmdb", "test.lmdb"):
        env_path = lmdb_dir / split
        if not env_path.exists():
            continue
        env = lmdb.open(str(env_path), subdir=True, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    key_str = key.decode("ascii")
                except Exception:
                    continue
                if not key_str.isdigit() and key_str != "length":
                    result.setdefault(key_str, value)
        env.close()
    return result


def build_scaffold_lmdb_splits(
    raw_dir: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    save_splits_path: Optional[str | Path] = None,
) -> SplitIndices:
    """Create scaffold-based LMDB splits from existing raw LMDBs.

    - Reads all entries from raw_dir/{train,valid,test}.lmdb
    - Computes scaffold split indices on the full SMILES list
    - Writes new LMDBs to out_dir/{train,valid,test}.lmdb
    - Optionally saves the indices JSON to save_splits_path
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_entries = _collect_all_entries_from_raw_lmdb(raw_dir)
    smiles = list(all_entries.keys())
    splits = scaffold_split_indices(smiles, frac=frac, seed=seed)

    if save_splits_path is not None:
        save_path = Path(save_splits_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(splits.to_json(), f, indent=2)

    # Materialize the three LMDBs
    train_entries = {smiles[i]: all_entries[smiles[i]] for i in splits.train}
    valid_entries = {smiles[i]: all_entries[smiles[i]] for i in splits.valid}
    test_entries = {smiles[i]: all_entries[smiles[i]] for i in splits.test}

    _write_lmdb_from_entries(out_dir / "train.lmdb", train_entries)
    _write_lmdb_from_entries(out_dir / "valid.lmdb", valid_entries)
    _write_lmdb_from_entries(out_dir / "test.lmdb", test_entries)

    return splits


def build_scaffold_lmdb_splits_from_dir(
    input_dir: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    save_splits_path: Optional[str | Path] = None,
    limit_total: Optional[int] = None,
) -> SplitIndices:
    """Create scaffold-based LMDB splits by copying entries from an existing LMDB dir.

    This avoids re-computation of features by reusing already-processed entries.
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_entries = _collect_all_entries_from_processed_lmdb_dir(input_dir)
    smiles = list(all_entries.keys())
    if limit_total is not None and limit_total > 0:
        smiles = smiles[: min(len(smiles), limit_total)]
    splits = scaffold_split_indices(smiles, frac=frac, seed=seed)

    if save_splits_path is not None:
        save_path = Path(save_splits_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(splits.to_json(), f, indent=2)

    train_entries = {smiles[i]: all_entries[smiles[i]] for i in splits.train}
    valid_entries = {smiles[i]: all_entries[smiles[i]] for i in splits.valid}
    test_entries = {smiles[i]: all_entries[smiles[i]] for i in splits.test}

    _write_lmdb_from_entries(out_dir / "train.lmdb", train_entries)
    _write_lmdb_from_entries(out_dir / "valid.lmdb", valid_entries)
    _write_lmdb_from_entries(out_dir / "test.lmdb", test_entries)

    return splits
