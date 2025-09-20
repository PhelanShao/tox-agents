#!/usr/bin/env python3
"""
Preprocess 3D conformers for ToxD4C dataset
This script pre-generates all 3D conformers and saves them to LMDB for fast training.
"""

import os
import sys
import lmdb
import pickle
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Fast3DConformerGenerator:
    """Fast 3D conformer generator with fallback strategies."""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        
    def generate_conformer(self, smiles: str) -> Optional[Dict]:
        """Generate 3D conformer for a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Sanitize molecule
            try:
                Chem.SanitizeMol(mol)
            except:
                return None
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D conformer
            conformer_generated = False
            
            # Strategy 1: ETKDG
            try:
                if AllChem.EmbedMolecule(mol, randomSeed=self.random_seed, useRandomCoords=True) != -1:
                    conformer_generated = True
            except:
                pass
            
            # Strategy 2: Basic embedding
            if not conformer_generated:
                try:
                    if AllChem.EmbedMolecule(mol, randomSeed=self.random_seed) != -1:
                        conformer_generated = True
                except:
                    pass
            
            if not conformer_generated:
                return None
            
            # Optimize geometry
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol):
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                else:
                    AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass
            
            # Extract features
            conformer = mol.GetConformer()
            coordinates = conformer.GetPositions().astype(np.float32)
            
            # Atom features
            atom_features = []
            for atom in mol.GetAtoms():
                features = self._get_atom_features(atom)
                atom_features.append(features)
            atom_features = np.array(atom_features, dtype=np.float32)
            
            # Bond features and edge indices
            bond_features = []
            edge_indices = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                edge_indices.extend([[i, j], [j, i]])
                
                bond_feat = self._get_bond_features(bond)
                bond_features.extend([bond_feat, bond_feat])
            
            if edge_indices:
                edge_index = np.array(edge_indices, dtype=np.int64).T
                bond_features = np.array(bond_features, dtype=np.float32)
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                bond_features = np.zeros((0, 10), dtype=np.float32)
            
            return {
                'atom_features': atom_features,
                'coordinates': coordinates,
                'bond_features': bond_features,
                'edge_index': edge_index,
                'num_atoms': len(atom_features),
                'smiles': smiles
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate conformer for {smiles}: {e}")
            return None
    
    def _get_atom_features(self, atom) -> List[float]:
        """Extract atom features."""
        features = []
        
        # Atomic number
        atomic_num = atom.GetAtomicNum()
        common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        features.extend([1.0 if atomic_num == x else 0.0 for x in common_atoms])
        
        # Degree
        degree = atom.GetDegree()
        features.extend([1.0 if degree == x else 0.0 for x in range(6)])
        
        # Formal charge
        formal_charge = atom.GetFormalCharge()
        features.extend([1.0 if formal_charge == x else 0.0 for x in [-2, -1, 0, 1, 2]])
        
        # Hybridization
        hybridization = atom.GetHybridization()
        hyb_types = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                     Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                     Chem.rdchem.HybridizationType.SP3D2]
        features.extend([1.0 if hybridization == x else 0.0 for x in hyb_types])
        
        # Additional features
        features.append(1.0 if atom.IsInRing() else 0.0)
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        
        # Total number of Hs
        num_hs = atom.GetTotalNumHs()
        features.extend([1.0 if num_hs == x else 0.0 for x in range(5)])
        
        return features
    
    def _get_bond_features(self, bond) -> List[float]:
        """Extract bond features."""
        features = []
        
        # Bond type
        bond_type = bond.GetBondType()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        features.extend([1.0 if bond_type == x else 0.0 for x in bond_types])
        
        # Conjugation
        features.append(1.0 if bond.GetIsConjugated() else 0.0)
        
        # Ring membership
        features.append(1.0 if bond.IsInRing() else 0.0)
        
        # Stereo
        stereo = bond.GetStereo()
        stereo_types = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
                        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE]
        features.extend([1.0 if stereo == x else 0.0 for x in stereo_types])
        
        return features

def process_smiles_batch(args):
    """Process a batch of SMILES strings."""
    smiles_batch, random_seed = args
    generator = Fast3DConformerGenerator(random_seed=random_seed)
    
    results = []
    for smiles in smiles_batch:
        result = generator.generate_conformer(smiles)
        results.append((smiles, result))
    
    return results

def preprocess_lmdb_dataset(input_lmdb_path: str, output_lmdb_path: str, 
                           num_workers: int = 4, batch_size: int = 100):
    """Preprocess LMDB dataset to include pre-generated 3D conformers."""
    logger.info(f"🚀 Starting 3D conformer preprocessing")
    logger.info(f"   Input: {input_lmdb_path}")
    logger.info(f"   Output: {output_lmdb_path}")
    logger.info(f"   Workers: {num_workers}")
    
    # Read all SMILES from input LMDB
    logger.info("📖 Reading SMILES from input LMDB...")
    
    input_env = lmdb.open(input_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    smiles_data = []
    
    with input_env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            smiles = key.decode('ascii')
            data = pickle.loads(value)
            smiles_data.append((smiles, data))
    
    input_env.close()
    
    logger.info(f"📊 Found {len(smiles_data)} molecules")
    
    # Prepare batches for parallel processing
    smiles_list = [item[0] for item in smiles_data]
    batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]
    
    # Process in parallel
    logger.info(f"⚡ Processing {len(batches)} batches with {num_workers} workers...")
    
    start_time = time.time()
    
    with mp.Pool(num_workers) as pool:
        batch_args = [(batch, 42) for batch in batches]
        batch_results = list(tqdm(
            pool.imap(process_smiles_batch, batch_args),
            total=len(batches),
            desc="Processing batches"
        ))
    
    # Flatten results
    conformer_results = {}
    failed_count = 0
    
    for batch_result in batch_results:
        for smiles, conformer_data in batch_result:
            if conformer_data is not None:
                conformer_results[smiles] = conformer_data
            else:
                failed_count += 1
    
    processing_time = time.time() - start_time
    success_rate = len(conformer_results) / len(smiles_data) * 100
    
    logger.info(f"✅ 3D conformer generation completed in {processing_time:.1f}s")
    logger.info(f"   Success rate: {success_rate:.1f}% ({len(conformer_results)}/{len(smiles_data)})")
    logger.info(f"   Failed: {failed_count}")
    
    # Write to output LMDB
    logger.info("💾 Writing preprocessed data to output LMDB...")
    
    output_env = lmdb.open(output_lmdb_path, map_size=10**12, lock=False)  # 1TB max size
    
    with output_env.begin(write=True) as txn:
        for smiles, original_data in tqdm(smiles_data, desc="Writing to LMDB"):
            if smiles in conformer_results:
                # Combine original data with 3D conformer data
                combined_data = original_data.copy()
                combined_data.update(conformer_results[smiles])
                
                txn.put(smiles.encode('ascii'), pickle.dumps(combined_data))
    
    output_env.close()
    
    logger.info(f"🎉 Preprocessing completed! Output saved to: {output_lmdb_path}")
    
    return {
        'total_molecules': len(smiles_data),
        'successful_conformers': len(conformer_results),
        'failed_conformers': failed_count,
        'success_rate': success_rate,
        'processing_time': processing_time
    }

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess 3D conformers for ToxD4C')
    parser.add_argument('--input_dir', type=str, default='data/data/dataset',
                       help='Input LMDB directory')
    parser.add_argument('--output_dir', type=str, default='data/data/dataset_3d',
                       help='Output LMDB directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for parallel processing')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = ['train', 'valid', 'test']
    
    for dataset in datasets:
        input_path = input_dir / f"{dataset}.lmdb"
        output_path = output_dir / f"{dataset}.lmdb"
        
        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset} dataset")
        logger.info(f"{'='*60}")
        
        result = preprocess_lmdb_dataset(
            str(input_path), 
            str(output_path),
            num_workers=args.num_workers,
            batch_size=args.batch_size
        )
        
        logger.info(f"📊 {dataset} dataset results:")
        for key, value in result.items():
            logger.info(f"   {key}: {value}")

if __name__ == "__main__":
    main()
