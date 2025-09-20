import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Fragments
from rdkit.Chem import rdFingerprintGenerator
import warnings

warnings.filterwarnings('ignore')


class FingerprintEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 fingerprint_type: str = 'ECFP',
                 hidden_dims: List[int] = [512, 256],
                 dropout: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        self.fingerprint_type = fingerprint_type
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MolecularDescriptorCalculator:
    def __init__(self):
        self.descriptor_functions = {
            'MW': Descriptors.MolWt,
            'LogP': Descriptors.MolLogP,
            'HBD': Descriptors.NumHDonors,
            'HBA': Descriptors.NumHAcceptors,
            'TPSA': Descriptors.TPSA,
            'NumRotatableBonds': Descriptors.NumRotatableBonds,
            'NumAromaticRings': Descriptors.NumAromaticRings,
            'FractionCSP3': Descriptors.FractionCSP3,
            'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
            'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles,
            'NumAliphaticRings': Descriptors.NumAliphaticRings,
            'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
            'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles,
            'RingCount': Descriptors.RingCount,
            'BertzCT': Descriptors.BertzCT,
        }
    
    def calculate_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        descriptors = []
        for desc_name, desc_func in self.descriptor_functions.items():
            try:
                value = desc_func(mol)
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                descriptors.append(value)
            except:
                descriptors.append(0.0)
        
        return np.array(descriptors)
    
    def get_descriptor_names(self) -> List[str]:
        return list(self.descriptor_functions.keys())


class FingerprintCalculator:
    def __init__(self):
        self.calculator = MolecularDescriptorCalculator()
    
    def calculate_ecfp(self, mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        try:
            fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = fp_gen.GetFingerprintAsBitVect(mol)
            return np.array(fp)
        except:
            return np.zeros(n_bits)
    
    def calculate_maccs(self, mol: Chem.Mol) -> np.ndarray:
        try:
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            return np.array(fp)
        except:
            return np.zeros(167)
    
    def calculate_rdkit_fp(self, mol: Chem.Mol, fp_size: int = 2048) -> np.ndarray:
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=fp_size)
            return np.array(fp)
        except:
            return np.zeros(fp_size)
    
    def calculate_avalon_fp(self, mol: Chem.Mol, n_bits: int = 512) -> np.ndarray:
        try:
            from rdkit.Avalon import pyAvalonTools
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
            return np.array(fp)
        except:
            return np.zeros(n_bits)
    
    def calculate_atom_pair_fp(self, mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
        try:
            fp = rdMolDescriptors.GetAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            return np.array(fp)
        except:
            return np.zeros(n_bits)


class MolecularFingerprintModule(nn.Module):
    def __init__(self, 
                 output_dim: int = 512,
                 fingerprint_configs: Optional[Dict] = None,
                 use_attention_fusion: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.output_dim = output_dim
        self.use_attention_fusion = use_attention_fusion
        
        if fingerprint_configs is None:
            fingerprint_configs = {
                'ecfp': {'n_bits': 2048, 'radius': 2},
                'maccs': {'n_bits': 167},
                'rdkit_fp': {'n_bits': 2048},
                'avalon': {'n_bits': 512},
                'atom_pair': {'n_bits': 2048},
                'descriptors': {'n_features': 15}
            }
        
        self.fingerprint_configs = fingerprint_configs
        
        self.fingerprint_encoders = nn.ModuleDict()
        for fp_name, config in fingerprint_configs.items():
            input_dim = config.get('n_bits', config.get('n_features', 512))
            self.fingerprint_encoders[fp_name] = FingerprintEncoder(
                input_dim=input_dim,
                output_dim=output_dim // len(fingerprint_configs),
                fingerprint_type=fp_name,
                dropout=dropout
            )
        
        total_fp_dim = output_dim
        
        if use_attention_fusion:
            encoded_fp_dim = output_dim // len(fingerprint_configs)
            self.attention_weights = nn.Sequential(
                nn.Linear(total_fp_dim, total_fp_dim // 4),
                nn.ReLU(),
                nn.Linear(total_fp_dim // 4, len(fingerprint_configs)),
                nn.Softmax(dim=-1)
            )
            
            self.feature_transform = nn.Sequential(
                nn.Linear(encoded_fp_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.feature_fusion = nn.Sequential(
                nn.Linear(total_fp_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        self.fp_calculator = None
    
    def _get_fp_calculator(self):
        if self.fp_calculator is None:
            self.fp_calculator = FingerprintCalculator()
        return self.fp_calculator
    
    def calculate_fingerprints_from_smiles(self, smiles: List[str]) -> Dict[str, torch.Tensor]:
        calculator = self._get_fp_calculator()
        
        batch_fingerprints = {fp_name: [] for fp_name in self.fingerprint_configs.keys()}
        
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    for fp_name, config in self.fingerprint_configs.items():
                        input_dim = config.get('n_bits', config.get('n_features', 512))
                        batch_fingerprints[fp_name].append(np.zeros(input_dim))
                    continue
                
                for fp_name, config in self.fingerprint_configs.items():
                    if fp_name == 'ecfp':
                        fp = calculator.calculate_ecfp(mol, 
                                                     radius=config.get('radius', 2),
                                                     n_bits=config.get('n_bits', 2048))
                    elif fp_name == 'maccs':
                        fp = calculator.calculate_maccs(mol)
                    elif fp_name == 'rdkit_fp':
                        fp = calculator.calculate_rdkit_fp(mol, fp_size=config.get('n_bits', 2048))
                    elif fp_name == 'avalon':
                        fp = calculator.calculate_avalon_fp(mol, n_bits=config.get('n_bits', 512))
                    elif fp_name == 'atom_pair':
                        fp = calculator.calculate_atom_pair_fp(mol, n_bits=config.get('n_bits', 2048))
                    elif fp_name == 'descriptors':
                        fp = calculator.calculator.calculate_descriptors(mol)
                    else:
                        input_dim = config.get('n_bits', config.get('n_features', 512))
                        fp = np.zeros(input_dim)
                    
                    batch_fingerprints[fp_name].append(fp)
                    
            except Exception as e:
                for fp_name, config in self.fingerprint_configs.items():
                    input_dim = config.get('n_bits', config.get('n_features', 512))
                    batch_fingerprints[fp_name].append(np.zeros(input_dim))
        
        tensor_fingerprints = {}
        for fp_name, fp_list in batch_fingerprints.items():
            tensor_fingerprints[fp_name] = torch.FloatTensor(np.array(fp_list))
        
        return tensor_fingerprints
    
    def forward(self, 
                fingerprints: Optional[Dict[str, torch.Tensor]] = None,
                smiles: Optional[List[str]] = None) -> torch.Tensor:
        if fingerprints is None and smiles is not None:
            fingerprints = self.calculate_fingerprints_from_smiles(smiles)
        elif fingerprints is None:
            raise ValueError("Either 'fingerprints' or 'smiles' must be provided.")
        
        device = next(self.parameters()).device
        for fp_name in fingerprints:
            fingerprints[fp_name] = fingerprints[fp_name].to(device)
        
        encoded_fingerprints = []
        fp_names = []
        
        for fp_name, fp_tensor in fingerprints.items():
            if fp_name in self.fingerprint_encoders:
                encoded_fp = self.fingerprint_encoders[fp_name](fp_tensor)
                encoded_fingerprints.append(encoded_fp)
                fp_names.append(fp_name)
        
        if not encoded_fingerprints:
            raise ValueError("No valid fingerprint features found.")
        
        if self.use_attention_fusion and len(encoded_fingerprints) > 1:
            concatenated_fps = torch.cat(encoded_fingerprints, dim=-1)
            
            attention_weights = self.attention_weights(concatenated_fps)
            
            weighted_fps = []
            start_idx = 0
            for i, encoded_fp in enumerate(encoded_fingerprints):
                end_idx = start_idx + encoded_fp.size(-1)
                weight = attention_weights[:, i:i+1]
                weighted_fp = encoded_fp * weight
                weighted_fps.append(weighted_fp)
                start_idx = end_idx
            
            summed_fps = torch.stack(weighted_fps, dim=0).sum(dim=0)
            final_features = self.feature_transform(summed_fps)
        else:
            concatenated_fps = torch.cat(encoded_fingerprints, dim=-1)
            final_features = self.feature_fusion(concatenated_fps)
        
        return final_features
    
    def get_fingerprint_importance(self, 
                                 fingerprints: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if not self.use_attention_fusion:
            num_fps = len(fingerprints)
            return {fp_name: 1.0/num_fps for fp_name in fingerprints.keys()}
        
        device = next(self.parameters()).device
        for fp_name in fingerprints:
            fingerprints[fp_name] = fingerprints[fp_name].to(device)
        
        encoded_fingerprints = []
        fp_names = []
        
        for fp_name, fp_tensor in fingerprints.items():
            if fp_name in self.fingerprint_encoders:
                encoded_fp = self.fingerprint_encoders[fp_name](fp_tensor)
                encoded_fingerprints.append(encoded_fp)
                fp_names.append(fp_name)
        
        concatenated_fps = torch.cat(encoded_fingerprints, dim=-1)
        attention_weights = self.attention_weights(concatenated_fps)
        
        avg_weights = attention_weights.mean(dim=0).cpu().detach().numpy()
        
        return {fp_names[i]: float(avg_weights[i]) for i in range(len(fp_names))}


if __name__ == "__main__":
    fp_module = MolecularFingerprintModule(
        output_dim=512,
        use_attention_fusion=True
    )
    
    test_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "CC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)C",
        "CCO"
    ]
    
    features = fp_module(smiles=test_smiles)
    print(f"Output feature shape: {features.shape}")
    
    fingerprints = fp_module.calculate_fingerprints_from_smiles(test_smiles)
    importance = fp_module.get_fingerprint_importance(fingerprints)
    print("Fingerprint Importance:")
    for fp_name, weight in importance.items():
        print(f"{fp_name}: {weight:.4f}")