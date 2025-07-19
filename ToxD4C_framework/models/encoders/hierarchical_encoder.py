import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv


class FunctionalGroupEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.functional_groups = {
            'benzene': '[c1ccccc1]',
            'carboxyl': '[CX3](=O)[OX2H1]',
            'hydroxyl': '[OX2H]',
            'amino': '[NX3;H2,H1;!$(NC=O)]',
            'carbonyl': '[CX3]=[OX1]',
            'ester': '[#6][CX3](=O)[OX2H0][#6]',
            'ether': '[OD2]([#6])[#6]',
            'amide': '[CX3](=[OX1])[NX3H2]',
            'nitro': '[NX3+](=O)[O-]',
            'sulfhydryl': '[SH]',
            'phosphate': '[PX4](=[OX1])[OX2H,OX1-]',
            'halogen': '[F,Cl,Br,I]',
        }
        
        self.fg_embeddings = nn.ModuleDict({
            name: nn.Linear(1, embed_dim // len(self.functional_groups))
            for name in self.functional_groups.keys()
        })
        
        self.fg_aggregator = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def identify_functional_groups(self, mol) -> Dict[str, int]:
        if mol is None:
            return {name: 0 for name in self.functional_groups.keys()}
        
        fg_counts = {}
        for name, smarts in self.functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                fg_counts[name] = len(matches)
            else:
                fg_counts[name] = 0
        
        return fg_counts
    
    def forward(self, mol_batch: List) -> torch.Tensor:
        batch_size = len(mol_batch)
        fg_features = []
        
        for mol in mol_batch:
            fg_counts = self.identify_functional_groups(mol)
            
            fg_embeds = []
            for name, count in fg_counts.items():
                count_tensor = torch.tensor([float(count)], device=next(self.parameters()).device)
                embed = self.fg_embeddings[name](count_tensor)
                fg_embeds.append(embed)
            
            mol_fg_features = torch.cat(fg_embeds, dim=0)
            fg_features.append(mol_fg_features)
        
        fg_batch = torch.stack(fg_features, dim=0)
        
        output = self.fg_aggregator(fg_batch)
        output = self.dropout(output)
        
        return output


class ScaffoldEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.scaffold_features = nn.Sequential(
            nn.Linear(10, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
    def extract_scaffold_features(self, mol) -> np.ndarray:
        if mol is None:
            return np.zeros(10)
        
        features = []
        try:
            features.append(mol.GetNumAtoms())
            features.append(mol.GetNumBonds())
            features.append(mol.GetNumHeavyAtoms())
            features.append(Descriptors.NumAromaticRings(mol))
            features.append(Descriptors.NumAliphaticRings(mol))
            features.append(Descriptors.RingCount(mol))
            features.append(Descriptors.FractionCsp3(mol) or 0)
            features.append(Descriptors.HeavyAtomCount(mol))
            features.append(Descriptors.NumRotatableBonds(mol))
            features.append(Descriptors.TPSA(mol))
            
        except Exception as e:
            print(f"Error calculating scaffold features: {e}")
            features = [0.0] * 10
        
        return np.array(features, dtype=np.float32)
    
    def forward(self, mol_batch: List) -> torch.Tensor:
        batch_features = []
        
        for mol in mol_batch:
            scaffold_feat = self.extract_scaffold_features(mol)
            batch_features.append(scaffold_feat)
        
        batch_tensor = torch.tensor(np.stack(batch_features), 
                                  device=next(self.parameters()).device)
        
        output = self.scaffold_features(batch_tensor)
        
        return output


class AtomicLevelEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, max_atoms: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_atoms = max_atoms
        
        self.atom_feature_dim = 9
        
        self.atom_encoder = nn.Sequential(
            nn.Linear(self.atom_feature_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        self.pos_encoder = nn.Parameter(torch.randn(max_atoms, embed_dim))
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, atom_features: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms, _ = atom_features.shape
        
        atom_embeds = self.atom_encoder(atom_features)
        
        pos_embeds = self.pos_encoder[:num_atoms].unsqueeze(0).expand(batch_size, -1, -1)
        atom_embeds = atom_embeds + pos_embeds
        
        attn_mask = ~atom_mask.bool()
        atom_embeds, _ = self.self_attention(
            atom_embeds, atom_embeds, atom_embeds,
            key_padding_mask=attn_mask
        )
        
        atom_embeds = self.norm(atom_embeds)
        
        mask_expanded = atom_mask.unsqueeze(-1).float()
        masked_embeds = atom_embeds * mask_expanded
        
        atom_counts = atom_mask.sum(dim=1, keepdim=True).float()
        global_repr = masked_embeds.sum(dim=1) / (atom_counts + 1e-8)
        
        return global_repr


class HierarchicalEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 hierarchy_levels: List[int],
                 dropout: float = 0.1):
        super().__init__()
        
        self.hierarchy_levels = hierarchy_levels
        self.num_hierarchies = len(hierarchy_levels)
        
        self.level_encoders = nn.ModuleList()
        current_dim = input_dim
        
        for num_layers in hierarchy_levels:
            level_encoder = self._create_gcn_block(current_dim, hidden_dim, num_layers, dropout)
            self.level_encoders.append(level_encoder)
            current_dim = hidden_dim
            
        self.fusion_layer = nn.Linear(hidden_dim * self.num_hierarchies, hidden_dim)
        self.fusion_activation = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)

    def _create_gcn_block(self, 
                          in_channels: int, 
                          out_channels: int, 
                          num_layers: int, 
                          dropout: float) -> nn.Module:
        layers = nn.ModuleList()
        
        layers.append(GCNConv(in_channels, out_channels))
        
        for _ in range(num_layers - 1):
            layers.append(GCNConv(out_channels, out_channels))
        
        activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])
        batch_norms = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_layers)])
        
        return nn.ModuleDict({
            'convs': layers,
            'activations': activations,
            'norms': batch_norms
        })

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        level_representations = []
        current_x = x
        
        for i, level_encoder in enumerate(self.level_encoders):
            level_x = current_x
            for j, conv in enumerate(level_encoder['convs']):
                level_x = conv(level_x, edge_index)
                level_x = level_encoder['norms'][j](level_x)
                level_x = level_encoder['activations'][j](level_x)
            
            graph_rep = global_mean_pool(level_x, batch)
            level_representations.append(graph_rep)
            
            current_x = level_x
            
        fused_rep = torch.cat(level_representations, dim=1)
        
        fused_rep = self.fusion_layer(fused_rep)
        fused_rep = self.fusion_activation(fused_rep)
        fused_rep = self.fusion_dropout(fused_rep)
        
        return fused_rep


class HierarchicalChemicalEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, max_atoms: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.atomic_encoder = AtomicLevelEncoder(embed_dim // 4, max_atoms)
        self.fg_encoder = FunctionalGroupEncoder(embed_dim // 4)
        self.scaffold_encoder = ScaffoldEncoder(embed_dim // 4)
        
        self.molecular_encoder = nn.Sequential(
            nn.Linear(15, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embed_dim // 4,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def extract_molecular_features(self, mol) -> np.ndarray:
        if mol is None:
            return np.zeros(15)
        
        features = []
        try:
            features.append(Descriptors.MolWt(mol))
            features.append(Descriptors.MolLogP(mol))
            features.append(Descriptors.NumHDonors(mol))
            features.append(Descriptors.NumHAcceptors(mol))
            features.append(Descriptors.TPSA(mol))
            features.append(Descriptors.NumRotatableBonds(mol))
            features.append(Descriptors.FractionCsp3(mol) or 0)
            features.append(Descriptors.HeavyAtomCount(mol))
            features.append(Descriptors.RingCount(mol))
            features.append(Descriptors.AromaticProportion(mol))
            features.append(Descriptors.BalabanJ(mol))
            features.append(Descriptors.BertzCT(mol))
            features.append(Descriptors.Chi0(mol))
            features.append(Descriptors.Chi1(mol))
            features.append(Descriptors.Kappa1(mol))
            
        except Exception as e:
            print(f"Error calculating molecular features: {e}")
            features = [0.0] * 15
            
        return np.array(features, dtype=np.float32)
    
    def forward(self, 
                atom_features: torch.Tensor,
                atom_mask: torch.Tensor,
                mol_batch: List,
                fingerprints: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = len(mol_batch)
        
        atomic_repr = self.atomic_encoder(atom_features, atom_mask)
        
        fg_repr = self.fg_encoder(mol_batch)
        
        scaffold_repr = self.scaffold_encoder(mol_batch)
        
        mol_features = []
        for mol in mol_batch:
            mol_feat = self.extract_molecular_features(mol)
            mol_features.append(mol_feat)
        
        mol_feat_tensor = torch.tensor(np.stack(mol_features),
                                     device=atom_features.device)
        molecular_repr = self.molecular_encoder(mol_feat_tensor)
        
        level_reprs = torch.stack([
            atomic_repr,
            fg_repr, 
            scaffold_repr,
            molecular_repr
        ], dim=1)
        
        fused_reprs, attention_weights = self.fusion_attention(
            level_reprs, level_reprs, level_reprs
        )
        
        final_repr = fused_reprs.view(batch_size, -1)
        final_repr = self.final_fusion(final_repr)
        final_repr = self.layer_norm(final_repr)
        
        return final_repr


if __name__ == "__main__":
    encoder = HierarchicalChemicalEncoder(embed_dim=512, max_atoms=256)
    
    batch_size = 4
    max_atoms = 256
    atom_feat_dim = 9
    
    atom_features = torch.randn(batch_size, max_atoms, atom_feat_dim)
    atom_mask = torch.ones(batch_size, max_atoms)
    
    mol_batch = [None] * batch_size
    
    try:
        output = encoder(atom_features, atom_mask, mol_batch)
        print(f"Output shape: {output.shape}")
        print(f"Hierarchical chemical encoder created successfully!")
    except Exception as e:
        print(f"Error: {e}")