"""
ToxD4C inference script.

Provides batch predictions for the 31 toxicity endpoints supported by the
model.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
from pathlib import Path
from rdkit import Chem
import warnings
from torch.utils.data import Dataset

# Ensure the project root is importable when running as a script
sys.path.append(str(Path(__file__).parent))
import argparse
from configs.toxd4c_config import (
    get_enhanced_toxd4c_config, CLASSIFICATION_TASKS, REGRESSION_TASKS
)
from data.lmdb_dataset import create_lmdb_dataloaders, MolecularFeatureExtractor, collate_lmdb_batch
from models.toxd4c import ToxD4C

warnings.filterwarnings('ignore')


class ToxD4CPredictor:
    """Utility that loads a trained ToxD4C model for inference."""
    
    def __init__(self, model_path: str, config: Dict, device: str = 'cpu'):
        self.device = device
        self.config = config
        
        # Instantiate the model
        self.model = ToxD4C(config=self.config, device=device).to(device)

        # Load parameters and move to eval mode
        self.load_model(model_path)
        self.model.eval()

        print("ToxD4C predictor loaded")
        print(f"Device: {device}")
        print(f"Checkpoint: {model_path}")

    def load_model(self, model_path: str):
        """Load model weights from disk."""
        if not Path(model_path).exists():
            print(f"Error: checkpoint not found {model_path}")
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint: {model_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {str(e)}")
            raise e

    def predict_on_loader(self, dataloader: torch.utils.data.DataLoader) -> pd.DataFrame:
        """Run batched inference over the provided dataloader."""
        all_smiles = []
        all_cls_preds = []
        all_reg_preds = []

        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                # Move tensors to the configured device
                data = {
                    'atom_features': batch['atom_features'].to(self.device),
                    'edge_index': batch['edge_index'].to(self.device),
                    'coordinates': batch['coordinates'].to(self.device),
                    'batch': batch['batch'].to(self.device)
                }
                smiles_list = batch['smiles']

                # Forward pass
                outputs = self.model(data, smiles_list)
                cls_preds = outputs['predictions']['classification']
                reg_preds = outputs['predictions']['regression']

                # Collect results
                all_smiles.extend(smiles_list)
                all_cls_preds.append(torch.sigmoid(cls_preds).cpu().numpy())
                all_reg_preds.append(reg_preds.cpu().numpy())

        # Skip saving if nothing was processed
        if not all_smiles:
            return pd.DataFrame()

        cls_preds_np = np.concatenate(all_cls_preds, axis=0)
        reg_preds_np = np.concatenate(all_reg_preds, axis=0)
        
        # Build a dataframe of results
        results_df = pd.DataFrame({'SMILES': all_smiles})

        # Attach classification probabilities and binary decisions
        for i, task_name in enumerate(CLASSIFICATION_TASKS):
            results_df[f"{task_name}_prob"] = cls_preds_np[:, i]
            results_df[task_name] = (cls_preds_np[:, i] > 0.5).astype(int)

        # Attach regression outputs
        for i, task_name in enumerate(REGRESSION_TASKS):
            results_df[task_name] = reg_preds_np[:, i]

        return results_df
    
class SmilesDataset(Dataset):
    """Dataset wrapper for plain-text SMILES lists.

    Supports tab- or space-delimited rows by taking the first column and skips
    empty lines or headers containing the word ``smiles``.
    """
    def __init__(self, smiles_list: List[str]):
        cleaned = []
        for raw in smiles_list:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if 'smiles' in low and (low.startswith('smiles') or low.split()[0] == 'smiles'):
                continue
            parts = line.split('\t') if ('\t' in line) else line.split()
            if not parts:
                continue
            smiles = parts[0].strip()
            if smiles:
                cleaned.append(smiles)

        self.smiles_list = cleaned
        self.feature_extractor = MolecularFeatureExtractor()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: failed to parse SMILES: {smiles}")
            return None

        graph_data = self.feature_extractor.mol_to_graph(mol)
        if graph_data is None:
            print(f"Warning: unable to generate graph features for SMILES: {smiles}")
            return None
            

        return {
            'atom_features': torch.tensor(graph_data['atom_features'], dtype=torch.float32),
            'bond_features': torch.tensor(graph_data['bond_features'], dtype=torch.float32),
            'edge_index': torch.tensor(graph_data['edge_index'], dtype=torch.long),
            'coordinates': torch.tensor(graph_data['coordinates'], dtype=torch.float32),
            'classification_labels': torch.zeros(len(CLASSIFICATION_TASKS)),
            'regression_labels': torch.zeros(len(REGRESSION_TASKS)),
            'classification_mask': torch.ones(len(CLASSIFICATION_TASKS), dtype=torch.bool),
            'regression_mask': torch.ones(len(REGRESSION_TASKS), dtype=torch.bool),
            'smiles': smiles
        }

def main():
    """Entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='ToxD4C inference script')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints_real/toxd4c_real_best.pth',
                        help='Path to the trained checkpoint file')
    parser.add_argument('--smiles_file', type=str, default=None,
                        help='Optional text file with SMILES strings (one per line)')
    parser.add_argument('--data_dir', type=str, default='data/dataset',
                        help='LMDB directory to use when --smiles_file is not provided')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help='Force the inference device (defaults to auto-detect)')
    parser.add_argument('--output_file', type=str, default='inference_results.csv',
                        help='Destination CSV for predictions')
    
    args = parser.parse_args()

    print("=== ToxD4C Inference ===")

    # Select device
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load a safe default config for inference
    config = get_enhanced_toxd4c_config()

    # Build the evaluation dataloader
    if args.smiles_file:
        print(f"Loading SMILES from file: {args.smiles_file}")
        try:
            with open(args.smiles_file, 'r') as f:
                smiles_list = f.readlines()

            dataset = SmilesDataset(smiles_list)
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_lmdb_batch
            )
            print(f"Loaded {len(dataset)} SMILES")
        except Exception as e:
            print(f"Failed to load SMILES file: {e}")
            return
    else:
        print(f"Loading data from LMDB directory: {args.data_dir}")
        try:
            _, _, test_loader = create_lmdb_dataloaders(
                args.data_dir,
                batch_size=args.batch_size
            )
            print(f"Loaded evaluation split from: {args.data_dir}")
            print(f"Number of batches: {len(test_loader)}")
        except Exception as e:
            print(f"Failed to load evaluation data: {e}")
            return

    # Construct the predictor
    try:
        predictor = ToxD4CPredictor(
            model_path=args.model_path,
            config=config,
            device=device
        )
    except Exception as e:
        print(f"Failed to initialise predictor: {e}")
        return

    # Run inference
    print("Running inference...")
    results_df = predictor.predict_on_loader(test_loader)

    # Persist results
    if not results_df.empty:
        results_df.to_csv(args.output_file, index=False)
        print(f"Predictions saved to: {args.output_file}")

        # Show a small preview
        print("\nPreview:")
        print(results_df.head())
    else:
        print("No predictions were generated.")

if __name__ == "__main__":
    main()
