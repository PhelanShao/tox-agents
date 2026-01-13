"""
Dataset class for multi-conformer LMDB data.
Directly loads 11 conformers from original dataset where coordinates is a list of arrays.
"""

import os
import sys
import lmdb
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_scaffold(smiles: str) -> Optional[str]:
    """Get Murcko scaffold for a molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return None


def scaffold_split_indices(smiles_list: List[str], train_ratio: float = 0.8,
                           valid_ratio: float = 0.1, seed: int = 42) -> Dict[str, List[int]]:
    """
    Split molecules by scaffold into train/valid/test sets.

    Args:
        smiles_list: List of SMILES strings
        train_ratio: Ratio for training set
        valid_ratio: Ratio for validation set
        seed: Random seed

    Returns:
        Dictionary with 'train', 'valid', 'test' keys containing indices
    """
    np.random.seed(seed)

    # Group molecules by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = get_scaffold(smiles)
        if scaffold is None:
            scaffold = f"_no_scaffold_{idx}"
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffolds by size (largest first) for more balanced splits
    scaffolds = list(scaffold_to_indices.items())
    np.random.shuffle(scaffolds)
    scaffolds.sort(key=lambda x: len(x[1]), reverse=True)

    n_total = len(smiles_list)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_indices = []
    valid_indices = []
    test_indices = []

    for scaffold, indices in scaffolds:
        if len(train_indices) < n_train:
            train_indices.extend(indices)
        elif len(valid_indices) < n_valid:
            valid_indices.extend(indices)
        else:
            test_indices.extend(indices)

    logger.info(f"Scaffold split: train={len(train_indices)}, valid={len(valid_indices)}, test={len(test_indices)}")
    logger.info(f"Number of unique scaffolds: {len(scaffolds)}")

    return {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices
    }


class OriginalMultiConformerDataset(Dataset):
    """
    Dataset that directly loads multi-conformer data from original LMDB.
    Original data format: coordinates is a list of 11 numpy arrays.

    This creates 11x samples by expanding each molecule into 11 conformer samples.
    """

    def __init__(self, lmdb_path: str, max_atoms: int = 64,
                 conformer_mode: str = 'all', n_conformers: int = 11,
                 sample_n_conformers: int = None):
        """
        Args:
            lmdb_path: Path to original LMDB database
            max_atoms: Maximum number of atoms
            conformer_mode:
                - 'all': Return all conformers as separate samples (11x data)
                - 'random': Randomly select one conformer per molecule each time
                - 'random_n': Randomly select n conformers per molecule (n specified by sample_n_conformers)
                - 'first': Always use the first conformer (conf_idx=0)
            n_conformers: Number of conformers in original data (default: 11)
            sample_n_conformers: Number of conformers to randomly sample when mode='random_n' (e.g., 3)
        """
        self.lmdb_path = lmdb_path
        self.max_atoms = max_atoms
        self.conformer_mode = conformer_mode
        self.n_conformers = n_conformers
        self.sample_n_conformers = sample_n_conformers if sample_n_conformers else n_conformers

        subdir_flag = os.path.isdir(lmdb_path)
        self.env = lmdb.open(
            lmdb_path,
            subdir=subdir_flag,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256
        )

        # Collect all SMILES keys
        self.smiles_keys = []
        logger.info("Collecting SMILES keys from original dataset...")
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                try:
                    key_str = key.decode('ascii')
                    if not key_str.isdigit() and key_str != 'length':
                        self.smiles_keys.append(key_str)
                except:
                    continue

        logger.info(f"Loaded LMDB: {lmdb_path}")
        logger.info(f"Unique molecules: {len(self.smiles_keys)}")
        logger.info(f"Conformer mode: {conformer_mode}")
        if conformer_mode == 'random_n':
            logger.info(f"Sampling {self.sample_n_conformers} conformers per molecule")

        # Build index: (smiles_idx, conformer_idx)
        if conformer_mode == 'all':
            self.index = []
            for smiles_idx in range(len(self.smiles_keys)):
                for conf_idx in range(n_conformers):
                    self.index.append((smiles_idx, conf_idx))
            logger.info(f"Total samples (all conformers): {len(self.index)}")
        elif conformer_mode == 'random_n':
            # For random_n mode, expand by sample_n_conformers
            self.index = []
            for smiles_idx in range(len(self.smiles_keys)):
                for _ in range(self.sample_n_conformers):
                    self.index.append((smiles_idx, None))  # None means random selection
            logger.info(f"Total samples (random {self.sample_n_conformers} conformers): {len(self.index)}")
        else:
            self.index = [(i, None) for i in range(len(self.smiles_keys))]
            logger.info(f"Total samples (per molecule): {len(self.index)}")

        # Feature extractor for atom/bond features
        self._init_feature_extractor()

    def _init_feature_extractor(self):
        """Initialize feature dimensions."""
        self.atom_features_dim = 119
        self.bond_features_dim = 12

    def __len__(self):
        return len(self.index)

    def _get_atom_features(self, atom) -> np.ndarray:
        """Extract atom features."""
        features = []

        # Atomic number one-hot
        atomic_num = atom.GetAtomicNum()
        one_hot = [0] * 118
        if 1 <= atomic_num <= 118:
            one_hot[atomic_num - 1] = 1
        features.extend(one_hot)

        # Pad to atom_features_dim
        while len(features) < self.atom_features_dim:
            features.append(0)

        return np.array(features[:self.atom_features_dim], dtype=np.float32)

    def _get_bond_features(self, bond) -> np.ndarray:
        """Extract bond features."""
        features = []

        # Bond type
        bond_type = bond.GetBondType()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bt in bond_types:
            features.append(1 if bond_type == bt else 0)

        features.append(int(bond.GetIsConjugated()))
        features.append(int(bond.IsInRing()))

        # Pad to bond_features_dim
        while len(features) < self.bond_features_dim:
            features.append(0)

        return np.array(features[:self.bond_features_dim], dtype=np.float32)

    def __getitem__(self, idx):
        smiles_idx, conf_idx = self.index[idx]
        smiles = self.smiles_keys[smiles_idx]

        # Determine which conformer to use
        if self.conformer_mode == 'random' or self.conformer_mode == 'random_n':
            # For both random modes, randomly select a conformer
            conf_idx = np.random.randint(0, self.n_conformers)
        elif self.conformer_mode == 'first':
            conf_idx = 0
        # For 'all' mode, conf_idx is already set from index

        with self.env.begin() as txn:
            try:
                data_bytes = txn.get(smiles.encode('ascii'))
                if data_bytes is None:
                    return None

                data = pickle.loads(data_bytes)

                # Get coordinates - original data has list of conformers
                coords_list = data.get('coordinates', [])
                if isinstance(coords_list, list) and len(coords_list) > conf_idx:
                    coordinates = np.array(coords_list[conf_idx], dtype=np.float32)
                elif isinstance(coords_list, np.ndarray):
                    if coords_list.ndim == 3:  # (n_conf, n_atoms, 3)
                        coordinates = coords_list[conf_idx].astype(np.float32)
                    else:
                        coordinates = coords_list.astype(np.float32)
                else:
                    return None

                # Get atom symbols and create molecule for features
                atoms = data.get('atoms', [])
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                mol = Chem.AddHs(mol)

                # Generate atom features
                atom_features = []
                for atom in mol.GetAtoms():
                    atom_features.append(self._get_atom_features(atom))
                atom_features = np.array(atom_features, dtype=np.float32)

                # Generate bond features and edge index
                edge_indices = []
                bond_features = []
                for bond in mol.GetBonds():
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    edge_indices.extend([[i, j], [j, i]])
                    bf = self._get_bond_features(bond)
                    bond_features.extend([bf, bf])

                if edge_indices:
                    edge_index = np.array(edge_indices, dtype=np.int64).T
                    bond_features = np.array(bond_features, dtype=np.float32)
                else:
                    edge_index = np.empty((2, 0), dtype=np.int64)
                    bond_features = np.empty((0, self.bond_features_dim), dtype=np.float32)

                # Check atom count match
                if len(atom_features) != len(coordinates):
                    # Try to match by truncating or regenerating coords
                    if len(coordinates) > len(atom_features):
                        coordinates = coordinates[:len(atom_features)]
                    else:
                        return None

                # Labels
                classification_labels = np.array(
                    data.get('classification_target', [0]*26), dtype=np.float32
                )
                regression_labels = np.array(
                    data.get('regression_target', [0]*5), dtype=np.float32
                )

                cls_mask = (classification_labels != -10000)
                reg_mask = (regression_labels != -10000.0)

                classification_labels = np.where(cls_mask, classification_labels, 0.0)
                regression_labels = np.where(reg_mask, regression_labels, 0.0)

                return {
                    'atom_features': torch.tensor(atom_features, dtype=torch.float32),
                    'bond_features': torch.tensor(bond_features, dtype=torch.float32),
                    'edge_index': torch.tensor(edge_index, dtype=torch.long),
                    'coordinates': torch.tensor(coordinates, dtype=torch.float32),
                    'classification_labels': torch.tensor(classification_labels, dtype=torch.float32),
                    'regression_labels': torch.tensor(regression_labels, dtype=torch.float32),
                    'classification_mask': torch.tensor(cls_mask.astype(np.float32), dtype=torch.bool),
                    'regression_mask': torch.tensor(reg_mask.astype(np.float32), dtype=torch.bool),
                    'smiles': smiles,
                    'conformer_idx': conf_idx
                }

            except Exception as e:
                logger.warning(f"Error loading sample {idx} (smiles={smiles}, conf={conf_idx}): {e}")
                return None

    def close(self):
        self.env.close()


def collate_multi_conformer_batch(batch):
    """Collate function for multi-conformer batches."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    atom_features_list = []
    bond_features_list = []
    edge_indices_list = []
    coordinates_list = []
    batch_indices = []
    
    cls_labels_list = []
    reg_labels_list = []
    cls_mask_list = []
    reg_mask_list = []
    smiles_list = []
    
    atom_offset = 0
    
    for i, sample in enumerate(batch):
        atom_features_list.append(sample['atom_features'])
        bond_features_list.append(sample['bond_features'])
        coordinates_list.append(sample['coordinates'])
        
        edge_index = sample['edge_index'].clone() if isinstance(sample['edge_index'], torch.Tensor) \
                     else torch.tensor(sample['edge_index'])
        edge_index = edge_index + atom_offset
        edge_indices_list.append(edge_index)
        
        num_atoms = sample['atom_features'].shape[0]
        batch_indices.extend([i] * num_atoms)
        atom_offset += num_atoms
        
        cls_labels_list.append(sample['classification_labels'])
        reg_labels_list.append(sample['regression_labels'])
        cls_mask_list.append(sample['classification_mask'])
        reg_mask_list.append(sample['regression_mask'])
        smiles_list.append(sample['smiles'])
    
    return {
        'atom_features': torch.cat([f if isinstance(f, torch.Tensor) else torch.tensor(f) 
                                    for f in atom_features_list], dim=0),
        'bond_features': torch.cat([f if isinstance(f, torch.Tensor) else torch.tensor(f) 
                                    for f in bond_features_list], dim=0) if bond_features_list else torch.empty(0),
        'edge_index': torch.cat(edge_indices_list, dim=1) if edge_indices_list else torch.empty(2, 0, dtype=torch.long),
        'coordinates': torch.cat([c if isinstance(c, torch.Tensor) else torch.tensor(c) 
                                  for c in coordinates_list], dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'classification_labels': torch.stack(cls_labels_list),
        'regression_labels': torch.stack(reg_labels_list),
        'classification_mask': torch.stack(cls_mask_list),
        'regression_mask': torch.stack(reg_mask_list),
        'smiles': smiles_list
    }


def create_original_multi_conformer_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    conformer_mode: str = 'all',
    n_conformers: int = 11,
    sample_n_conformers: int = None,
    split_method: str = 'original',
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42
):
    """
    Create DataLoaders for multi-conformer training using original dataset format.

    Args:
        data_dir: Directory containing train.lmdb, valid.lmdb, test.lmdb or all_data.lmdb
        batch_size: Batch size
        num_workers: Number of data loading workers
        conformer_mode: 'all' (11x data), 'random' (sample 1 per mol), 'random_n' (sample n per mol), or 'first'
        n_conformers: Number of conformers in original data
        sample_n_conformers: Number of conformers to sample for 'random_n' mode (e.g., 3)
        split_method: 'original' (use existing splits) or 'scaffold' (scaffold-based split)
        train_ratio: Training set ratio (only for scaffold split)
        valid_ratio: Validation set ratio (only for scaffold split)
        seed: Random seed for scaffold split
    """
    if split_method == 'scaffold':
        # Use all_data.lmdb and perform scaffold split
        all_data_path = os.path.join(data_dir, 'all_data.lmdb')
        if not os.path.exists(all_data_path):
            # Try train.lmdb as fallback
            all_data_path = os.path.join(data_dir, 'train.lmdb')
            logger.info(f"all_data.lmdb not found, using train.lmdb for scaffold split")

        logger.info(f"Loading data from {all_data_path} for scaffold split...")

        # Create a temporary dataset to get all SMILES
        temp_dataset = OriginalMultiConformerDataset(
            all_data_path,
            conformer_mode='first',
            n_conformers=n_conformers
        )

        # Get all SMILES and perform scaffold split
        smiles_list = temp_dataset.smiles_keys
        split_indices = scaffold_split_indices(smiles_list, train_ratio, valid_ratio, seed)

        # Create datasets with filtered indices
        train_dataset = OriginalMultiConformerDatasetWithIndices(
            all_data_path,
            indices=split_indices['train'],
            smiles_keys=smiles_list,
            conformer_mode=conformer_mode,
            n_conformers=n_conformers,
            sample_n_conformers=sample_n_conformers
        )
        valid_dataset = OriginalMultiConformerDatasetWithIndices(
            all_data_path,
            indices=split_indices['valid'],
            smiles_keys=smiles_list,
            conformer_mode='first',
            n_conformers=n_conformers
        )
        test_dataset = OriginalMultiConformerDatasetWithIndices(
            all_data_path,
            indices=split_indices['test'],
            smiles_keys=smiles_list,
            conformer_mode='first',
            n_conformers=n_conformers
        )

        logger.info(f"Scaffold split created: train={len(train_dataset)}, valid={len(valid_dataset)}, test={len(test_dataset)}")

    else:
        # Use original split files
        train_dataset = OriginalMultiConformerDataset(
            os.path.join(data_dir, 'train.lmdb'),
            conformer_mode=conformer_mode,
            n_conformers=n_conformers,
            sample_n_conformers=sample_n_conformers
        )
        valid_dataset = OriginalMultiConformerDataset(
            os.path.join(data_dir, 'valid.lmdb'),
            conformer_mode='first',  # Always use first conformer for validation
            n_conformers=n_conformers
        )
        test_dataset = OriginalMultiConformerDataset(
            os.path.join(data_dir, 'test.lmdb'),
            conformer_mode='first',  # Always use first conformer for test
            n_conformers=n_conformers
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multi_conformer_batch,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multi_conformer_batch,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multi_conformer_batch,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader


class OriginalMultiConformerDatasetWithIndices(Dataset):
    """
    Dataset that loads multi-conformer data with specific molecule indices.
    Used for scaffold split where we need to select specific molecules.
    """

    def __init__(self, lmdb_path: str, indices: List[int], smiles_keys: List[str],
                 max_atoms: int = 64, conformer_mode: str = 'all',
                 n_conformers: int = 11, sample_n_conformers: int = None):
        """
        Args:
            lmdb_path: Path to LMDB database
            indices: List of molecule indices to include
            smiles_keys: List of all SMILES keys
            max_atoms: Maximum number of atoms
            conformer_mode: Conformer selection mode
            n_conformers: Number of conformers in data
            sample_n_conformers: Number of conformers to sample
        """
        self.lmdb_path = lmdb_path
        self.max_atoms = max_atoms
        self.conformer_mode = conformer_mode
        self.n_conformers = n_conformers
        self.sample_n_conformers = sample_n_conformers if sample_n_conformers else n_conformers

        # Store the selected SMILES keys
        self.smiles_keys = [smiles_keys[i] for i in indices]

        subdir_flag = os.path.isdir(lmdb_path)
        self.env = lmdb.open(
            lmdb_path,
            subdir=subdir_flag,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256
        )

        # Build index
        if conformer_mode == 'all':
            self.index = []
            for smiles_idx in range(len(self.smiles_keys)):
                for conf_idx in range(n_conformers):
                    self.index.append((smiles_idx, conf_idx))
        elif conformer_mode == 'random_n':
            self.index = []
            for smiles_idx in range(len(self.smiles_keys)):
                for _ in range(self.sample_n_conformers):
                    self.index.append((smiles_idx, None))
        else:
            self.index = [(i, None) for i in range(len(self.smiles_keys))]

        # Feature dimensions
        self.atom_features_dim = 119
        self.bond_features_dim = 12

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        smiles_idx, conf_idx = self.index[idx]
        smiles = self.smiles_keys[smiles_idx]

        # Determine conformer index
        if self.conformer_mode == 'random' or self.conformer_mode == 'random_n':
            conf_idx = np.random.randint(0, self.n_conformers)
        elif self.conformer_mode == 'first':
            conf_idx = 0

        try:
            with self.env.begin() as txn:
                data = pickle.loads(txn.get(smiles.encode('ascii')))

            if data is None:
                return None

            # Get coordinates for this conformer
            coords_list = data.get('coordinates', [])
            if isinstance(coords_list, list) and len(coords_list) > conf_idx:
                coords = np.array(coords_list[conf_idx], dtype=np.float32)
            else:
                coords = np.array(coords_list, dtype=np.float32)

            # Create molecule for features
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            mol = Chem.AddHs(mol)
            num_atoms = mol.GetNumAtoms()

            if num_atoms > self.max_atoms or num_atoms == 0:
                return None

            # Ensure coords match atom count
            if len(coords) != num_atoms:
                if len(coords) > num_atoms:
                    coords = coords[:num_atoms]
                else:
                    return None

            # Extract atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                atom_features.append(features)
            atom_features = np.array(atom_features, dtype=np.float32)

            # Build edge index and bond features
            edge_list = []
            bond_features = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_list.extend([[i, j], [j, i]])
                bf = self._get_bond_features(bond)
                bond_features.extend([bf, bf])

            if edge_list:
                edge_index = np.array(edge_list, dtype=np.int64).T
                bond_features = np.array(bond_features, dtype=np.float32)
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                bond_features = np.empty((0, self.bond_features_dim), dtype=np.float32)

            # Get labels and masks - try both naming conventions
            cls_labels = data.get('classification_labels', data.get('classification_target', np.zeros(26, dtype=np.float32)))
            reg_labels = data.get('regression_labels', data.get('regression_target', np.zeros(5, dtype=np.float32)))

            cls_labels = np.array(cls_labels, dtype=np.float32)
            reg_labels = np.array(reg_labels, dtype=np.float32)

            # Create masks
            cls_mask = np.array(cls_labels != -10000, dtype=np.float32)
            reg_mask = np.array(reg_labels != -10000, dtype=np.float32)

            # Replace invalid values
            cls_labels = np.where(cls_labels == -10000, 0, cls_labels).astype(np.float32)
            reg_labels = np.where(reg_labels == -10000, 0, reg_labels).astype(np.float32)

            return {
                'atom_features': torch.from_numpy(atom_features),
                'bond_features': torch.from_numpy(bond_features),
                'edge_index': torch.from_numpy(edge_index),
                'coordinates': torch.from_numpy(coords),
                'classification_labels': torch.from_numpy(cls_labels),
                'regression_labels': torch.from_numpy(reg_labels),
                'classification_mask': torch.tensor(cls_mask, dtype=torch.bool),
                'regression_mask': torch.tensor(reg_mask, dtype=torch.bool),
                'smiles': smiles,
                'conformer_idx': conf_idx
            }

        except Exception as e:
            return None

    def _get_atom_features(self, atom) -> np.ndarray:
        """Extract atom features (same as original class)."""
        features = []

        # Atomic number one-hot (118 dim)
        atomic_num = atom.GetAtomicNum()
        one_hot = [0] * 118
        if 1 <= atomic_num <= 118:
            one_hot[atomic_num - 1] = 1
        features.extend(one_hot)

        # Pad to atom_features_dim (119)
        while len(features) < self.atom_features_dim:
            features.append(0)

        return np.array(features[:self.atom_features_dim], dtype=np.float32)

    def _get_bond_features(self, bond) -> np.ndarray:
        """Extract bond features."""
        features = []

        # Bond type (4 dim)
        bond_type = bond.GetBondType()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bt in bond_types:
            features.append(1 if bond_type == bt else 0)

        # Is conjugated (1 dim)
        features.append(int(bond.GetIsConjugated()))

        # Is in ring (1 dim)
        features.append(int(bond.IsInRing()))

        # Pad to bond_features_dim
        while len(features) < self.bond_features_dim:
            features.append(0)

        return np.array(features[:self.bond_features_dim], dtype=np.float32)
