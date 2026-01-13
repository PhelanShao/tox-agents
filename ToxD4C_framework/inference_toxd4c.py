import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import sys
from pathlib import Path
from rdkit import Chem
import warnings
from torch.utils.data import Dataset
import json
from scipy.spatial.distance import cdist

sys.path.append(str(Path(__file__).parent))
import argparse
from configs.toxd4c_config import (
    get_enhanced_toxd4c_config, CLASSIFICATION_TASKS, REGRESSION_TASKS
)
from data.lmdb_dataset import create_lmdb_dataloaders, MolecularFeatureExtractor, collate_lmdb_batch
from models.toxd4c import ToxD4C

warnings.filterwarnings('ignore')


class ToxD4CPredictor:
    def __init__(self, model_path: str, config: Dict, device: str = 'cpu'):
        self.device = device
        self.config = config
        
        self.model = ToxD4C(config=self.config, device=device).to(device)
        
        self.load_model(model_path)
        self.model.eval()
        
        print(f"ToxD4C Predictor loaded.")
        print(f"Device: {device}")
        print(f"Model: {model_path}")
    
    def load_model(self, model_path: str):
        if not Path(model_path).exists():
            print(f"Error: Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded model from: {model_path}")
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            raise e
    
    def predict_on_loader(self, dataloader: torch.utils.data.DataLoader) -> Tuple[pd.DataFrame, List[Dict]]:
        all_identifiers = []
        all_cls_preds = []
        all_reg_preds = []
        all_interpretations = []

        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                data = {
                    'atom_features': batch['atom_features'].to(self.device),
                    'edge_index': batch['edge_index'].to(self.device),
                    'coordinates': batch['coordinates'].to(self.device),
                    'batch': batch['batch'].to(self.device)
                }
                identifiers = batch.get('identifier', batch.get('smiles'))
                
                # The model's fingerprint module expects smiles, but for XYZ it's not available.
                # We pass the identifier list, but the module should handle it gracefully.
                outputs = self.model(data, identifiers)
                cls_preds = outputs['predictions']['classification']
                reg_preds = outputs['predictions']['regression']
                
                all_identifiers.extend(identifiers)
                all_cls_preds.append(torch.sigmoid(cls_preds).cpu().numpy())
                all_reg_preds.append(reg_preds.cpu().numpy())

                interp_batch = outputs.get('interpretation', {})
                uncertainties_batch = outputs.get('uncertainties', {})
                
                for i in range(len(identifiers)):
                    sample_interp = {'identifier': identifiers[i]}
                    if interp_batch.get('main_encoder') and interp_batch['main_encoder'].get('attention_weights') is not None:
                        sample_interp['attention_weights'] = interp_batch['main_encoder']['attention_weights'][i].cpu().numpy().tolist()
                    if interp_batch.get('main_encoder') and interp_batch['main_encoder'].get('fusion_weights') is not None:
                        sample_interp['fusion_weights'] = interp_batch['main_encoder']['fusion_weights'][i].cpu().numpy().tolist()
                    if interp_batch.get('fingerprint_attention') is not None:
                        sample_interp['fingerprint_attention'] = interp_batch['fingerprint_attention'][i].cpu().numpy().tolist()
                    
                    sample_uncertainty = {}
                    for task, uncert_tensor in uncertainties_batch.items():
                        sample_uncertainty[task] = uncert_tensor[i].cpu().numpy().tolist()
                    sample_interp['uncertainties'] = sample_uncertainty
                    
                    all_interpretations.append(sample_interp)

        if not all_identifiers:
            return pd.DataFrame(), []

        cls_preds_np = np.concatenate(all_cls_preds, axis=0)
        reg_preds_np = np.concatenate(all_reg_preds, axis=0)
        
        results_df = pd.DataFrame({'Identifier': all_identifiers})
        
        for i, task_name in enumerate(CLASSIFICATION_TASKS):
            results_df[task_name] = (cls_preds_np[:, i] > 0.5).astype(int)
            
        for i, task_name in enumerate(REGRESSION_TASKS):
            results_df[task_name] = reg_preds_np[:, i]
            
        return results_df, all_interpretations
    
class SmilesDataset(Dataset):
    def __init__(self, smiles_list: List[str]):
        self.smiles_list = [s.strip() for s in smiles_list if s.strip()]
        self.feature_extractor = MolecularFeatureExtractor()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            return None
        
        graph_data = self.feature_extractor.mol_to_graph(mol)
        if graph_data is None:
            print(f"Warning: Could not generate graph features for SMILES: {smiles}")
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
            'smiles': smiles,
            'identifier': smiles
        }

class XYZDataset(Dataset):
    def __init__(self, xyz_file: str):
        self.feature_extractor = MolecularFeatureExtractor()
        self.frames = self._parse_xyz(xyz_file)

    def _parse_xyz(self, xyz_file):
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        
        frames = []
        i = 0
        while i < len(lines):
            try:
                num_atoms = int(lines[i].strip())
                frame_data = {
                    'title': lines[i+1].strip(),
                    'atoms': [],
                    'coords': []
                }
                for j in range(num_atoms):
                    parts = lines[i+2+j].strip().split()
                    frame_data['atoms'].append(parts[0])
                    frame_data['coords'].append([float(p) for p in parts[1:4]])
                frames.append(frame_data)
                i += num_atoms + 2
            except (ValueError, IndexError):
                i += 1
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        atom_symbols = frame['atoms']
        coords = np.array(frame['coords'], dtype=np.float32)

        mol = self._mol_from_xyz(atom_symbols, coords)
        if mol is None:
            print(f"Warning: Could not build molecule from XYZ frame {idx}")
            return None

        # Now, extract all features from the *reconstructed* mol object
        # to ensure consistency.
        
        # Atom features
        atom_features = np.array([self.feature_extractor.get_atom_features(atom) for atom in mol.GetAtoms()])

        # Bond features and edge index
        bond_features = []
        edge_indices = []
        for bond in mol.GetBonds():
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

        # The coordinates are from the original XYZ, but we use the atom count from the RDKit mol
        final_coords = mol.GetConformer().GetPositions()

        return {
            'atom_features': torch.tensor(atom_features, dtype=torch.float32),
            'bond_features': torch.tensor(bond_features, dtype=torch.float32),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'coordinates': torch.tensor(final_coords, dtype=torch.float32),
            'classification_labels': torch.zeros(len(CLASSIFICATION_TASKS)),
            'regression_labels': torch.zeros(len(REGRESSION_TASKS)),
            'classification_mask': torch.ones(len(CLASSIFICATION_TASKS), dtype=torch.bool),
            'regression_mask': torch.ones(len(REGRESSION_TASKS), dtype=torch.bool),
            'identifier': f"XYZ_Frame_{idx}"
        }

    def _mol_from_xyz(self, symbols, coords):
        mol = Chem.RWMol()
        for sym in symbols:
            mol.AddAtom(Chem.Atom(sym))
        
        conf = Chem.Conformer(len(symbols))
        for i in range(len(symbols)):
            conf.SetAtomPosition(i, coords[i].tolist())
        mol.AddConformer(conf)

        dist_matrix = cdist(coords, coords)
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                r1 = Chem.GetPeriodicTable().GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
                r2 = Chem.GetPeriodicTable().GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
                if dist_matrix[i, j] < (r1 + r2) * 0.6:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
        
        try:
            Chem.SanitizeMol(mol)
            return mol.GetMol()
        except:
            return None


def collate_xyz_batch(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    atom_features_list = []
    edge_indices_list = []
    coordinates_list = []
    batch_indices = []
    identifiers_list = []
    
    atom_offset = 0
    
    for i, sample in enumerate(batch):
        atom_features_list.append(sample['atom_features'])
        coordinates_list.append(sample['coordinates'])
        
        edge_index = sample['edge_index'] + atom_offset
        edge_indices_list.append(edge_index)
        
        num_atoms = sample['atom_features'].shape[0]
        batch_indices.extend([i] * num_atoms)
        atom_offset += num_atoms
        
        identifiers_list.append(sample['identifier'])
    
    return {
        'atom_features': torch.cat(atom_features_list, dim=0),
        'edge_index': torch.cat(edge_indices_list, dim=1),
        'coordinates': torch.cat(coordinates_list, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'identifier': identifiers_list,
        'smiles': identifiers_list # Add this for compatibility
    }


def main():
    parser = argparse.ArgumentParser(description='ToxD4C Inference Script')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints_real/toxd4c_real_best.pth',
                        help='Path to the trained model')
    parser.add_argument('--smiles_file', type=str, default=None, help='Input file containing SMILES strings')
    parser.add_argument('--xyz_file', type=str, default=None, help='Input file in XYZ format')
    parser.add_argument('--data_dir', type=str, default='data/dataset', help='LMDB data directory (if no input file is provided)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--output_file', type=str, default='inference_results.csv', help='Output CSV file for predictions')
    
    args = parser.parse_args()

    print("=== ToxD4C Inference ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = get_enhanced_toxd4c_config()
    
    test_loader = None
    if args.smiles_file:
        print(f"Loading data from SMILES file: {args.smiles_file}")
        try:
            with open(args.smiles_file, 'r') as f:
                smiles_list = f.readlines()
            
            dataset = SmilesDataset(smiles_list)
            print(f"Successfully loaded {len(dataset)} SMILES.")
            test_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_lmdb_batch
            )
        except Exception as e:
            print(f"Failed to load data from SMILES file: {e}")
            return
    elif args.xyz_file:
        print(f"Loading data from XYZ file: {args.xyz_file}")
        try:
            dataset = XYZDataset(args.xyz_file)
            print(f"Successfully loaded {len(dataset)} frames from XYZ file.")
            test_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_xyz_batch
            )
        except Exception as e:
            print(f"Failed to load data from XYZ file: {e}")
            return
    else:
        print(f"Loading data from LMDB directory: {args.data_dir}")
        try:
            _, _, test_loader = create_lmdb_dataloaders(
                args.data_dir,
                batch_size=args.batch_size
            )
            print(f"Successfully loaded test data from: {args.data_dir}")
            print(f"Number of test batches: {len(test_loader)}")
        except Exception as e:
            print(f"Failed to load test data: {e}")
            return

    if test_loader is None:
        print("No data loader created. Exiting.")
        return

    try:
        predictor = ToxD4CPredictor(
            model_path=args.model_path,
            config=config,
            device=device
        )
    except Exception as e:
        print(f"Failed to create predictor: {e}")
        return

    print("Starting prediction...")
    results_df, interpretations = predictor.predict_on_loader(test_loader)
    
    if not results_df.empty:
        results_df.to_csv(args.output_file, index=False)
        print(f"Prediction results saved to: {args.output_file}")
        
        interp_path = Path(args.output_file).with_suffix('.json')
        with open(interp_path, 'w') as f:
            json.dump(interpretations, f, indent=2)
        print(f"Interpretation data saved to: {interp_path}")

        print("\nPrediction results preview:")
        print(results_df.head())
    else:
        print("No predictions were generated.")

if __name__ == "__main__":
    main()