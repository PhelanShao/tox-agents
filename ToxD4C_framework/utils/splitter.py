#!/usr/bin/env python3
"""
Data splitting utilities for ToxD4C
Addresses A1 reviewer concerns about data splitting strategies.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
import warnings

logger = logging.getLogger(__name__)

class MolecularSplitter:
    """
    Comprehensive molecular data splitting strategies for robust evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def random_split(self, 
                    smiles_list: List[str], 
                    train_size: float = 0.8, 
                    val_size: float = 0.1,
                    stratify_labels: Optional[np.ndarray] = None) -> Tuple[List[int], List[int], List[int]]:
        """
        Random split with optional stratification.
        
        Args:
            smiles_list: List of SMILES strings
            train_size: Fraction for training set
            val_size: Fraction for validation set
            stratify_labels: Labels for stratification (optional)
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        n_samples = len(smiles_list)
        indices = np.arange(n_samples)
        
        test_size = 1.0 - train_size - val_size
        
        if stratify_labels is not None:
            # Stratified split
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_size, 
                stratify=stratify_labels, random_state=self.random_state
            )
            
            # Further split train_val into train and val
            train_val_labels = stratify_labels[train_val_idx]
            relative_val_size = val_size / (train_size + val_size)
            
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=relative_val_size,
                stratify=train_val_labels, random_state=self.random_state
            )
        else:
            # Simple random split
            train_val_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=self.random_state
            )
            
            relative_val_size = val_size / (train_size + val_size)
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=relative_val_size, random_state=self.random_state
            )
        
        logger.info(f"Random split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
    
    def scaffold_split(self, 
                      smiles_list: List[str], 
                      train_size: float = 0.8, 
                      val_size: float = 0.1,
                      include_chirality: bool = False) -> Tuple[List[int], List[int], List[int]]:
        """
        Scaffold-based split using Murcko scaffolds.
        
        Args:
            smiles_list: List of SMILES strings
            train_size: Fraction for training set
            val_size: Fraction for validation set
            include_chirality: Whether to include chirality in scaffold
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        logger.info("Performing scaffold split...")
        
        # Generate scaffolds
        scaffolds = defaultdict(list)
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                continue
            
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)
                scaffolds[scaffold_smiles].append(i)
            except Exception as e:
                logger.warning(f"Error generating scaffold for {smiles}: {e}")
                # Assign to a unique scaffold
                scaffolds[f"error_{i}"].append(i)
        
        # Sort scaffolds by size (largest first)
        scaffold_sets = list(scaffolds.values())
        scaffold_sets.sort(key=len, reverse=True)
        
        # Distribute scaffolds to splits
        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count, test_count = 0, 0, 0
        n_total = len(smiles_list)
        
        for scaffold_set in scaffold_sets:
            # Decide which split to add this scaffold to
            train_frac = train_count / n_total if n_total > 0 else 0
            val_frac = val_count / n_total if n_total > 0 else 0
            test_frac = test_count / n_total if n_total > 0 else 0
            
            if train_frac < train_size:
                train_idx.extend(scaffold_set)
                train_count += len(scaffold_set)
            elif val_frac < val_size:
                val_idx.extend(scaffold_set)
                val_count += len(scaffold_set)
            else:
                test_idx.extend(scaffold_set)
                test_count += len(scaffold_set)
        
        logger.info(f"Scaffold split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        logger.info(f"Unique scaffolds: {len(scaffolds)}")
        
        return train_idx, val_idx, test_idx
    
    def cluster_split(self, 
                     smiles_list: List[str], 
                     train_size: float = 0.8, 
                     val_size: float = 0.1,
                     n_clusters: Optional[int] = None,
                     fingerprint_type: str = 'morgan') -> Tuple[List[int], List[int], List[int]]:
        """
        Cluster-based split using molecular fingerprints.
        
        Args:
            smiles_list: List of SMILES strings
            train_size: Fraction for training set
            val_size: Fraction for validation set
            n_clusters: Number of clusters (auto if None)
            fingerprint_type: Type of fingerprint ('morgan', 'rdkit', 'maccs')
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        logger.info(f"Performing cluster split with {fingerprint_type} fingerprints...")
        
        # Generate fingerprints
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                if fingerprint_type == 'morgan':
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                elif fingerprint_type == 'rdkit':
                    fp = Chem.RDKFingerprint(mol)
                elif fingerprint_type == 'maccs':
                    fp = AllChem.GetMACCSKeysFingerprint(mol)
                else:
                    raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")
                
                fingerprints.append(np.array(fp))
                valid_indices.append(i)
                
            except Exception as e:
                logger.warning(f"Error generating fingerprint for {smiles}: {e}")
                continue
        
        if len(fingerprints) == 0:
            raise ValueError("No valid fingerprints generated")
        
        # Convert to numpy array
        fp_matrix = np.array(fingerprints)
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = max(10, len(valid_indices) // 50)  # Heuristic
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(fp_matrix)
        
        # Group indices by cluster
        clusters = defaultdict(list)
        for idx, cluster_id in zip(valid_indices, cluster_labels):
            clusters[cluster_id].append(idx)
        
        # Distribute clusters to splits (similar to scaffold split)
        cluster_sets = list(clusters.values())
        cluster_sets.sort(key=len, reverse=True)
        
        train_idx, val_idx, test_idx = [], [], []
        train_count, val_count, test_count = 0, 0, 0
        n_total = len(valid_indices)
        
        for cluster_set in cluster_sets:
            train_frac = train_count / n_total if n_total > 0 else 0
            val_frac = val_count / n_total if n_total > 0 else 0
            
            if train_frac < train_size:
                train_idx.extend(cluster_set)
                train_count += len(cluster_set)
            elif val_frac < val_size:
                val_idx.extend(cluster_set)
                val_count += len(cluster_set)
            else:
                test_idx.extend(cluster_set)
                test_count += len(cluster_set)
        
        logger.info(f"Cluster split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        logger.info(f"Number of clusters: {n_clusters}")
        
        return train_idx, val_idx, test_idx
    
    def temporal_split(self, 
                      smiles_list: List[str], 
                      timestamps: List[Union[str, pd.Timestamp]], 
                      train_size: float = 0.8, 
                      val_size: float = 0.1) -> Tuple[List[int], List[int], List[int]]:
        """
        Temporal split based on timestamps.
        
        Args:
            smiles_list: List of SMILES strings
            timestamps: List of timestamps
            train_size: Fraction for training set
            val_size: Fraction for validation set
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        logger.info("Performing temporal split...")
        
        # Convert timestamps to pandas datetime
        if isinstance(timestamps[0], str):
            timestamps = pd.to_datetime(timestamps)
        
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        n_total = len(smiles_list)
        
        # Calculate split points
        train_end = int(n_total * train_size)
        val_end = int(n_total * (train_size + val_size))
        
        train_idx = sorted_indices[:train_end].tolist()
        val_idx = sorted_indices[train_end:val_end].tolist()
        test_idx = sorted_indices[val_end:].tolist()
        
        logger.info(f"Temporal split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return train_idx, val_idx, test_idx
    
    def analyze_split_quality(self, 
                             smiles_list: List[str], 
                             train_idx: List[int], 
                             val_idx: List[int], 
                             test_idx: List[int]) -> Dict[str, float]:
        """
        Analyze the quality of a data split.
        
        Args:
            smiles_list: List of SMILES strings
            train_idx: Training set indices
            val_idx: Validation set indices
            test_idx: Test set indices
            
        Returns:
            Dictionary with split quality metrics
        """
        logger.info("Analyzing split quality...")
        
        # Calculate Tanimoto similarities
        def get_fingerprints(indices):
            fps = []
            for idx in indices:
                mol = Chem.MolFromSmiles(smiles_list[idx])
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
            return fps
        
        train_fps = get_fingerprints(train_idx)
        val_fps = get_fingerprints(val_idx)
        test_fps = get_fingerprints(test_idx)
        
        # Calculate average similarities
        def avg_similarity(fps1, fps2):
            if not fps1 or not fps2:
                return 0.0
            
            similarities = []
            for fp1 in fps1:
                sims = DataStructs.BulkTanimotoSimilarity(fp1, fps2)
                similarities.extend(sims)
            
            return np.mean(similarities) if similarities else 0.0
        
        train_val_sim = avg_similarity(train_fps, val_fps)
        train_test_sim = avg_similarity(train_fps, test_fps)
        val_test_sim = avg_similarity(val_fps, test_fps)
        
        # Calculate scaffold overlap
        def get_scaffolds(indices):
            scaffolds = set()
            for idx in indices:
                mol = Chem.MolFromSmiles(smiles_list[idx])
                if mol is not None:
                    try:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        scaffold_smiles = Chem.MolToSmiles(scaffold)
                        scaffolds.add(scaffold_smiles)
                    except:
                        continue
            return scaffolds
        
        train_scaffolds = get_scaffolds(train_idx)
        val_scaffolds = get_scaffolds(val_idx)
        test_scaffolds = get_scaffolds(test_idx)
        
        scaffold_overlap_train_val = len(train_scaffolds & val_scaffolds) / len(train_scaffolds | val_scaffolds) if train_scaffolds | val_scaffolds else 0
        scaffold_overlap_train_test = len(train_scaffolds & test_scaffolds) / len(train_scaffolds | test_scaffolds) if train_scaffolds | test_scaffolds else 0
        scaffold_overlap_val_test = len(val_scaffolds & test_scaffolds) / len(val_scaffolds | test_scaffolds) if val_scaffolds | test_scaffolds else 0
        
        quality_metrics = {
            'train_val_similarity': train_val_sim,
            'train_test_similarity': train_test_sim,
            'val_test_similarity': val_test_sim,
            'scaffold_overlap_train_val': scaffold_overlap_train_val,
            'scaffold_overlap_train_test': scaffold_overlap_train_test,
            'scaffold_overlap_val_test': scaffold_overlap_val_test,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx)
        }
        
        logger.info("Split quality analysis completed")
        for metric, value in quality_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return quality_metrics

def create_external_validation_splits(toxcast_smiles: List[str], 
                                     tox21_smiles: List[str],
                                     internal_smiles: List[str]) -> Dict[str, List[int]]:
    """
    Create external validation splits using ToxCast and Tox21 data.
    
    Args:
        toxcast_smiles: ToxCast SMILES
        tox21_smiles: Tox21 SMILES  
        internal_smiles: Internal dataset SMILES
        
    Returns:
        Dictionary with split assignments
    """
    logger.info("Creating external validation splits...")
    
    # Find overlaps
    toxcast_set = set(toxcast_smiles)
    tox21_set = set(tox21_smiles)
    internal_set = set(internal_smiles)
    
    splits = {
        'internal_only': [],
        'toxcast_overlap': [],
        'tox21_overlap': [],
        'both_overlap': []
    }
    
    for i, smiles in enumerate(internal_smiles):
        in_toxcast = smiles in toxcast_set
        in_tox21 = smiles in tox21_set
        
        if in_toxcast and in_tox21:
            splits['both_overlap'].append(i)
        elif in_toxcast:
            splits['toxcast_overlap'].append(i)
        elif in_tox21:
            splits['tox21_overlap'].append(i)
        else:
            splits['internal_only'].append(i)
    
    logger.info(f"External validation splits created:")
    for split_name, indices in splits.items():
        logger.info(f"  {split_name}: {len(indices)} compounds")
    
    return splits

# Example usage
if __name__ == "__main__":
    # Test the splitter
    splitter = MolecularSplitter(random_state=42)
    
    # Example SMILES
    test_smiles = [
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "CCN(CC)CC",  # triethylamine
        "CC(C)O"  # isopropanol
    ]
    
    # Test different splitting strategies
    print("Random split:")
    train, val, test = splitter.random_split(test_smiles)
    print(f"Train: {train}, Val: {val}, Test: {test}")
    
    print("\nScaffold split:")
    train, val, test = splitter.scaffold_split(test_smiles)
    print(f"Train: {train}, Val: {val}, Test: {test}")
    
    # Analyze split quality
    quality = splitter.analyze_split_quality(test_smiles, train, val, test)
    print(f"\nSplit quality: {quality}")
