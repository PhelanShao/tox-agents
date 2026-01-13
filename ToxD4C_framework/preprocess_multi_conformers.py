"""
Preprocess LMDB data with multiple conformers per molecule.
Generates 11 conformers for each molecule using RDKit's ETKDG method.
"""

import os
import sys
import lmdb
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(str(Path(__file__).parent))
from data.lmdb_dataset import MolecularFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiConformerGenerator:
    """Generate multiple 3D conformers for a molecule."""
    
    def __init__(self, n_conformers: int = 11, random_seed: int = 42):
        self.n_conformers = n_conformers
        self.random_seed = random_seed
        self.feature_extractor = MolecularFeatureExtractor()
    
    def generate_conformers(self, mol) -> Optional[List[np.ndarray]]:
        """
        Generate multiple conformers for a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of coordinate arrays, one per conformer
        """
        if mol is None:
            return None
        
        try:
            mol = Chem.AddHs(mol)
            
            # Use ETKDG v3 for conformer generation
            params = AllChem.ETKDGv3()
            params.randomSeed = self.random_seed
            params.numThreads = 0  # Use all available threads
            params.pruneRmsThresh = 0.5  # Prune similar conformers
            
            # Generate multiple conformers
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=self.n_conformers,
                params=params
            )
            
            if len(conf_ids) == 0:
                # Fallback: try with random coordinates
                params.useRandomCoords = True
                conf_ids = AllChem.EmbedMultipleConfs(
                    mol, 
                    numConfs=self.n_conformers,
                    params=params
                )
            
            if len(conf_ids) == 0:
                # Final fallback: generate single conformer
                if AllChem.EmbedMolecule(mol, randomSeed=self.random_seed) != -1:
                    conf_ids = [0]
                else:
                    return None
            
            # Optimize each conformer with force field
            conformer_coords = []
            for conf_id in conf_ids:
                try:
                    # Try MMFF first
                    if AllChem.MMFFHasAllMoleculeParams(mol):
                        AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
                    else:
                        # Fallback to UFF
                        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
                except Exception:
                    pass  # Use unoptimized conformer if optimization fails
                
                conformer = mol.GetConformer(conf_id)
                coords = conformer.GetPositions().astype(np.float32)
                conformer_coords.append(coords)
            
            # If we got fewer conformers than requested, duplicate to fill
            while len(conformer_coords) < self.n_conformers:
                # Add slightly perturbed versions of existing conformers
                idx = len(conformer_coords) % len(conf_ids)
                noise = np.random.normal(0, 0.05, conformer_coords[idx].shape).astype(np.float32)
                conformer_coords.append(conformer_coords[idx] + noise)
            
            return conformer_coords[:self.n_conformers]
            
        except Exception as e:
            logger.warning(f"Error generating conformers: {e}")
            return None
    
    def mol_to_graph_multi_conf(self, mol) -> Optional[Dict]:
        """
        Convert molecule to graph with multiple conformers.
        
        Returns dict with:
            - atom_features: (num_atoms, feature_dim)
            - bond_features: (num_edges, feature_dim)
            - edge_index: (2, num_edges)
            - coordinates: list of (num_atoms, 3) arrays, one per conformer
        """
        if mol is None:
            return None
        
        try:
            mol_h = Chem.AddHs(mol)
            
            # Generate atom features (same for all conformers)
            atom_features = []
            for atom in mol_h.GetAtoms():
                atom_features.append(self.feature_extractor.get_atom_features(atom))
            atom_features = np.array(atom_features, dtype=np.float32)
            
            # Generate bond features and edge indices (same for all conformers)
            bond_features = []
            edge_indices = []
            for bond in mol_h.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                bond_feat = self.feature_extractor.get_bond_features(bond)
                edge_indices.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
                bond_features.extend([bond_feat, bond_feat])
            
            if not edge_indices:
                edge_index = np.empty((2, 0), dtype=np.int64)
                bond_features = np.empty((0, self.feature_extractor.bond_features_dim), dtype=np.float32)
            else:
                edge_index = np.array(edge_indices, dtype=np.int64).T
                bond_features = np.array(bond_features, dtype=np.float32)
            
            # Generate multiple conformers
            conformer_coords = self.generate_conformers(mol)
            if conformer_coords is None:
                return None
            
            return {
                'atom_features': atom_features,
                'bond_features': bond_features,
                'edge_index': edge_index,
                'coordinates': conformer_coords,  # List of coordinate arrays
                'num_atoms': mol_h.GetNumAtoms(),
                'num_conformers': len(conformer_coords)
            }
            
        except Exception as e:
            logger.warning(f"Error in mol_to_graph_multi_conf: {e}")
            return None


def preprocess_multi_conformer_lmdb(input_db_path: str, output_db_path: str, 
                                     max_atoms: int = 64, n_conformers: int = 11):
    """
    Pre-process LMDB dataset to generate multiple conformers per molecule.
    """
    logger.info(f"Starting multi-conformer preprocessing.")
    logger.info(f"Input LMDB: {input_db_path}")
    logger.info(f"Output LMDB: {output_db_path}")
    logger.info(f"Number of conformers per molecule: {n_conformers}")

    Path(output_db_path).parent.mkdir(parents=True, exist_ok=True)

    generator = MultiConformerGenerator(n_conformers=n_conformers)

    subdir_flag = os.path.isdir(input_db_path)
    in_env = lmdb.open(input_db_path, subdir=subdir_flag, readonly=True, lock=False, 
                       readahead=False, meminit=False)
    
    map_size = 1024 * 1024 * 1024 * 100  # 100 GB for multi-conformer data
    out_env = lmdb.open(output_db_path, map_size=map_size)

    processed_count = 0
    error_count = 0
    total_conformers = 0

    with in_env.begin() as in_txn, out_env.begin(write=True) as out_txn:
        cursor = in_txn.cursor()
        
        for key, value in tqdm(cursor, desc="Generating multi-conformer data"):
            try:
                key_str = key.decode('ascii')
                if not key_str.isdigit() and key_str != 'length':
                    smiles = key_str
                    original_data = pickle.loads(value)

                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        error_count += 1
                        continue
                    
                    Chem.SanitizeMol(mol)

                    # Generate graph with multiple conformers
                    graph_data = generator.mol_to_graph_multi_conf(mol)
                    if graph_data is None:
                        error_count += 1
                        continue

                    if graph_data['num_atoms'] > max_atoms:
                        continue

                    # Store each conformer as a separate entry
                    for conf_idx, coords in enumerate(graph_data['coordinates']):
                        conf_key = f"{smiles}__conf_{conf_idx}"
                        
                        processed_sample = {
                            'atom_features': torch.tensor(graph_data['atom_features'], dtype=torch.float32),
                            'bond_features': torch.tensor(graph_data['bond_features'], dtype=torch.float32),
                            'edge_index': torch.tensor(graph_data['edge_index'], dtype=torch.long),
                            'coordinates': torch.tensor(coords, dtype=torch.float32),
                            'classification_target': original_data.get('classification_target', [0]*26),
                            'regression_target': original_data.get('regression_target', [0]*5),
                            'smiles': smiles,
                            'conformer_idx': conf_idx,
                            'num_conformers': n_conformers
                        }
                        
                        out_txn.put(conf_key.encode('ascii'), pickle.dumps(processed_sample))
                        total_conformers += 1
                    
                    processed_count += 1

            except Exception as e:
                logger.warning(f"Skipping molecule {key_str} due to error: {e}")
                error_count += 1
                continue
    
    logger.info("Multi-conformer preprocessing finished.")
    logger.info(f"Successfully processed: {processed_count} molecules.")
    logger.info(f"Total conformers generated: {total_conformers}")
    logger.info(f"Failed/skipped: {error_count} molecules.")

    in_env.close()
    out_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess LMDB with multiple conformers.')
    parser.add_argument('--data_dir', type=str, default='data/dataset', 
                        help='Directory containing raw LMDB files.')
    parser.add_argument('--output_dir', type=str, default='data/multi_conformer', 
                        help='Directory to save multi-conformer LMDB files.')
    parser.add_argument('--max_atoms', type=int, default=64, 
                        help='Maximum number of atoms per molecule.')
    parser.add_argument('--n_conformers', type=int, default=11, 
                        help='Number of conformers to generate per molecule.')
    
    args = parser.parse_args()

    for split in ['train', 'valid', 'test']:
        input_path = os.path.join(args.data_dir, f"{split}.lmdb")
        output_path = os.path.join(args.output_dir, f"{split}.lmdb")
        
        if os.path.exists(input_path):
            preprocess_multi_conformer_lmdb(
                input_path, output_path, 
                args.max_atoms, args.n_conformers
            )
        else:
            logger.warning(f"Input file not found, skipping: {input_path}")

