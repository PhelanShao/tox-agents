import os
import sys
import lmdb
import pickle
import logging
from pathlib import Path
from typing import Dict, List
import argparse
from tqdm import tqdm

import torch
from rdkit import Chem

# Add project path
sys.path.append(str(Path(__file__).parent))
from data.lmdb_dataset import MolecularFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_lmdb(input_db_path: str, output_db_path: str, max_atoms: int):
    """
    Pre-processes an LMDB dataset to generate and store graph features, including 3D coordinates.
    This avoids re-computing these features on-the-fly during training.
    """
    logger.info(f"Starting preprocessing.")
    logger.info(f"Input LMDB: {input_db_path}")
    logger.info(f"Output LMDB: {output_db_path}")

    # Ensure output directory exists
    Path(output_db_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize feature extractor
    feature_extractor = MolecularFeatureExtractor()

    # Open input and output LMDB environments
    # Autodetect LMDB layout: file-backed (subdir=False) vs directory (subdir=True)
    subdir_flag = os.path.isdir(input_db_path)
    in_env = lmdb.open(input_db_path, subdir=subdir_flag, readonly=True, lock=False, readahead=False, meminit=False)
    # Set a large map_size for the output DB to avoid "MapFullError"
    map_size = 1024 * 1024 * 1024 * 50  # 50 GB
    out_env = lmdb.open(output_db_path, map_size=map_size)

    processed_count = 0
    error_count = 0

    with in_env.begin() as in_txn, out_env.begin(write=True) as out_txn:
        cursor = in_txn.cursor()
        
        # Use tqdm for a progress bar
        for key, value in tqdm(cursor, desc="Preprocessing molecules"):
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

                    # This is the time-consuming step
                    graph_data = feature_extractor.mol_to_graph(mol)
                    if graph_data is None:
                        error_count += 1
                        continue

                    # Truncate if necessary
                    if len(graph_data['atom_features']) > max_atoms:
                        continue

                    # Prepare the data to be saved
                    processed_sample = {
                        'atom_features': torch.tensor(graph_data['atom_features'], dtype=torch.float32),
                        'bond_features': torch.tensor(graph_data['bond_features'], dtype=torch.float32),
                        'edge_index': torch.tensor(graph_data['edge_index'], dtype=torch.long),
                        'coordinates': torch.tensor(graph_data['coordinates'], dtype=torch.float32),
                        'classification_target': original_data.get('classification_target', [0]*26),
                        'regression_target': original_data.get('regression_target', [0]*5),
                        'smiles': smiles
                    }
                    
                    # Serialize and write to the new database
                    out_txn.put(key, pickle.dumps(processed_sample))
                    processed_count += 1

            except Exception as e:
                logger.warning(f"Skipping molecule {key_str} due to error: {e}")
                error_count += 1
                continue
    
    logger.info("Preprocessing finished.")
    logger.info(f"Successfully processed: {processed_count} molecules.")
    logger.info(f"Failed/skipped: {error_count} molecules.")

    # Close environments
    in_env.close()
    out_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess raw LMDB data for faster training.')
    parser.add_argument('--data_dir', type=str, default='data/data/dataset', help='Directory containing the raw LMDB files (train.lmdb, valid.lmdb, test.lmdb).')
    parser.add_argument('--output_dir', type=str, default='data/data/processed', help='Directory to save the processed LMDB files.')
    parser.add_argument('--max_atoms', type=int, default=64, help='Maximum number of atoms to include per molecule.')
    
    args = parser.parse_args()

    for split in ['train', 'valid', 'test']:
        input_path = os.path.join(args.data_dir, f"{split}.lmdb")
        output_path = os.path.join(args.output_dir, f"{split}.lmdb")
        
        if os.path.exists(input_path):
            preprocess_lmdb(input_path, output_path, args.max_atoms)
        else:
            logger.warning(f"Input file not found, skipping: {input_path}")
