import os
import sys
import lmdb
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(str(Path(__file__).parent.parent))
from configs.toxd4c_config import get_enhanced_toxd4c_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LMDBToxD4CDataset(Dataset):
    def __init__(self, lmdb_path: str, max_atoms: int = 64):
        self.lmdb_path = lmdb_path
        self.max_atoms = max_atoms
        
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256
        )
        
        self.smiles_keys = []
        logger.info("Starting to collect SMILES keys...")
        count = 0
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                count += 1
                if count % 5000 == 0:
                    logger.info(f"Processed {count} entries...")
                
                try:
                    key_str = key.decode('ascii')
                    if not key_str.isdigit() and key_str != 'length':
                        self.smiles_keys.append(key_str)
                except:
                    continue
                
        
        logger.info(f"Loaded LMDB dataset: {lmdb_path}")
        logger.info(f"Found {len(self.smiles_keys)} molecules")
        
        self.atom_to_num = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54
        }
        
        self.feature_extractor = MolecularFeatureExtractor()
    
    def __len__(self):
        return len(self.smiles_keys)
    
    def __getitem__(self, idx):
        smiles = self.smiles_keys[idx]
        
        with self.env.begin() as txn:
            try:
                data_bytes = txn.get(smiles.encode('ascii'))
                if data_bytes is None:
                    return None

                data = pickle.loads(data_bytes)

                # For preprocessed data, all necessary tensors are already stored.
                classification_labels = np.array(data.get('classification_target', [0]*26), dtype=np.float32)
                regression_labels = np.array(data.get('regression_target', [0]*5), dtype=np.float32)

                cls_mask = (classification_labels != -10000)
                reg_mask = (regression_labels != -10000.0)
                
                classification_labels = np.where(cls_mask, classification_labels, 0.0)
                regression_labels = np.where(reg_mask, regression_labels, 0.0)

                return {
                    'atom_features': data['atom_features'],
                    'bond_features': data['bond_features'],
                    'edge_index': data['edge_index'],
                    'coordinates': data['coordinates'],
                    'classification_labels': torch.tensor(classification_labels, dtype=torch.float32),
                    'regression_labels': torch.tensor(regression_labels, dtype=torch.float32),
                    'classification_mask': torch.tensor(cls_mask.astype(np.float32), dtype=torch.bool),
                    'regression_mask': torch.tensor(reg_mask.astype(np.float32), dtype=torch.bool),
                    'smiles': data['smiles']
                }
                
            except Exception as e:
                logger.warning(f"Error loading preprocessed sample {idx} ({smiles}): {e}")
                return None
    
    def _create_simple_features(self, atoms):
        num_atoms = len(atoms)
        
        atom_features = np.zeros((num_atoms, 119), dtype=np.float32)
        for i, atom_num in enumerate(atoms):
            if 1 <= atom_num <= 118:
                atom_features[i, atom_num-1] = 1.0
        
        edge_index = []
        bond_features = []
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                edge_index.extend([[i, j], [j, i]])
                bond_features.extend([np.ones(12, dtype=np.float32), np.ones(12, dtype=np.float32)])
        
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            bond_features = [np.zeros(12, dtype=np.float32)]
        
        edge_index = np.array(edge_index).T
        bond_features = np.array(bond_features)
        
        return atom_features, bond_features, edge_index
    

class MolecularFeatureExtractor:
    def __init__(self):
        self.atom_features_dim = 119
        self.bond_features_dim = 12
        
    def get_atom_features(self, atom) -> np.ndarray:
        features = []
        
        atomic_num = atom.GetAtomicNum()
        features.extend(self._one_hot(atomic_num, list(range(1, 119))))
        
        degree = atom.GetDegree()
        features.extend(self._one_hot(degree, [0, 1, 2, 3, 4, 5]))
        
        formal_charge = atom.GetFormalCharge()
        features.extend(self._one_hot(formal_charge, [-1, 0, 1, 2]))
        
        hybridization = atom.GetHybridization()
        features.extend(self._one_hot(hybridization, [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]))
        
        features.append(int(atom.IsInRing()))
        
        features.append(int(atom.GetIsAromatic()))
        
        num_hs = atom.GetTotalNumHs()
        features.extend(self._one_hot(num_hs, [0, 1, 2, 3, 4]))
        
        while len(features) < self.atom_features_dim:
            features.append(0)
        
        return np.array(features[:self.atom_features_dim], dtype=np.float32)
    
    def get_bond_features(self, bond) -> np.ndarray:
        features = []
        
        bond_type = bond.GetBondType()
        features.extend(self._one_hot(bond_type, [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]))
        
        features.append(int(bond.GetIsConjugated()))
        
        features.append(int(bond.IsInRing()))
        
        stereo = bond.GetStereo()
        features.extend(self._one_hot(stereo, [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE
        ]))
        
        while len(features) < self.bond_features_dim:
            features.append(0)
        
        return np.array(features[:self.bond_features_dim], dtype=np.float32)
    
    def _one_hot(self, value, choices):
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding
    
    def mol_to_graph(self, mol) -> Dict[str, np.ndarray]:
        if mol is None:
            return None

        try:
            mol = Chem.AddHs(mol)
            
            if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
                return None
            
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception as e:
                logger.warning(f"Force field optimization failed, using unoptimized 3D conformation. Error: {e}")

            conformer = mol.GetConformer()
            coordinates = conformer.GetPositions()

        except Exception:
            return None

        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        atom_features = np.array(atom_features)
        
        bond_features = []
        edge_indices = []
        
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_feat = self.get_bond_features(bond)
            
            edge_indices.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
            bond_features.extend([bond_feat, bond_feat])
        
        if not edge_indices:
            edge_index = np.empty((2, 0), dtype=np.int64)
            bond_features = np.empty((0, self.bond_features_dim), dtype=np.float32)
        else:
            edge_index = np.array(edge_indices, dtype=np.int64).T
            bond_features = np.array(bond_features, dtype=np.float32)

        return {
            'atom_features': atom_features,
            'bond_features': bond_features,
            'edge_index': edge_index,
            'coordinates': coordinates,
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': len(bond_features)
        }


def collate_lmdb_batch(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    atom_features_list = []
    bond_features_list = []
    edge_indices_list = []
    coordinates_list = []
    batch_indices = []
    
    classification_labels_list = []
    regression_labels_list = []
    classification_masks_list = []
    regression_masks_list = []
    smiles_list = []
    
    atom_offset = 0
    
    for i, sample in enumerate(batch):
        atom_features_list.append(sample['atom_features'])
        bond_features_list.append(sample['bond_features'])
        coordinates_list.append(sample['coordinates'])
        
        edge_index = sample['edge_index'] + atom_offset
        edge_indices_list.append(edge_index)
        
        num_atoms = sample['atom_features'].shape[0]
        batch_indices.extend([i] * num_atoms)
        atom_offset += num_atoms
        
        classification_labels_list.append(sample['classification_labels'])
        regression_labels_list.append(sample['regression_labels'])
        classification_masks_list.append(sample['classification_mask'])
        regression_masks_list.append(sample['regression_mask'])
        smiles_list.append(sample['smiles'])
    
    batch_data = {
        'atom_features': torch.cat(atom_features_list, dim=0),
        'bond_features': torch.cat(bond_features_list, dim=0),
        'edge_index': torch.cat(edge_indices_list, dim=1),
        'coordinates': torch.cat(coordinates_list, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'classification_labels': torch.stack(classification_labels_list),
        'regression_labels': torch.stack(regression_labels_list),
        'classification_mask': torch.stack(classification_masks_list),
        'regression_mask': torch.stack(regression_masks_list),
        'smiles': smiles_list
    }
    
    return batch_data


def create_lmdb_dataloaders(data_dir: str, 
                           batch_size: int = 4,
                           max_atoms: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_path = Path(data_dir)
    
    train_file = data_path / "train.lmdb"
    valid_file = data_path / "valid.lmdb"
    test_file = data_path / "test.lmdb"
    
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not valid_file.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    logger.info(f"Using LMDB files:")
    logger.info(f"  Training set: {train_file}")
    logger.info(f"  Validation set: {valid_file}")
    logger.info(f"  Test set: {test_file}")
    
    train_dataset = LMDBToxD4CDataset(str(train_file), max_atoms)
    valid_dataset = LMDBToxD4CDataset(str(valid_file), max_atoms)
    test_dataset = LMDBToxD4CDataset(str(test_file), max_atoms)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_lmdb_batch
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_lmdb_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_lmdb_batch
    )
    
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    data_dir = "/mnt/backup3/toxscan/ToxScan/Toxd4c/Toxd4c/data/dataset"
    
    try:
        train_loader, valid_loader, test_loader = create_lmdb_dataloaders(data_dir, batch_size=2)
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(valid_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        
        for batch in train_loader:
            print("\nBatch data shapes:")
            print(f"Atom features: {batch['atom_features'].shape}")
            print(f"Bond features: {batch['bond_features'].shape}")
            print(f"Edge index: {batch['edge_index'].shape}")
            print(f"Coordinates: {batch['coordinates'].shape}")
            print(f"Batch index: {batch['batch'].shape}")
            print(f"Classification labels: {batch['classification_labels'].shape}")
            print(f"Regression labels: {batch['regression_labels'].shape}")
            print(f"Number of SMILES: {len(batch['smiles'])}")
            break
        
        print("\nLMDB data loader test successful!")
        
    except Exception as e:
        print(f"LMDB data loader test failed: {e}")