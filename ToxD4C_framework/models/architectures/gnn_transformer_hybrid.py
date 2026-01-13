import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, to_dense_batch

class GATLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0
        
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        alpha = torch.einsum('bic,bjc->bij', q, k) / math.sqrt(self.head_dim)
        
        adj = to_dense_adj(edge_index, max_num_nodes=x.size(1)).squeeze(0)
        alpha = alpha.masked_fill(adj == 0, -1e9)
        
        attention_weights = F.softmax(alpha, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.bmm(attention_weights, v)
        return output

class GraphAttentionNetwork(MessagePassing):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__(aggr='add', node_dim=0)
        self.num_layers = num_layers
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim, eps=1e-5))
            
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.input_projection(x)
        
        for i in range(self.num_layers):
            x_res = x
            x = self.propagate(edge_index, x=x, layer_idx=i)
            x = self.norm_layers[i](x + x_res)
            x = F.relu(x)
            
        dense_x, node_mask = to_dense_batch(x, batch)
        
        output = self.output_projection(dense_x)
        output = self.dropout(output)
        
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)
            
        return output

    def message(self, x_j, x_i, index, ptr, size_i, layer_idx):
        return x_j


class MolecularTransformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                node_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_projection(node_features)
        
        h = self.pos_encoding(h)
        
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        
        output = self.output_projection(h)
        output = self.dropout(output)
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class DynamicFusionModule(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.feature_enhance = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self,
                gnn_features: torch.Tensor,
                transformer_features: torch.Tensor,
                node_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        combined_features = torch.cat([gnn_features, transformer_features], dim=-1)
        fusion_weights = self.weight_generator(combined_features)
        
        enhanced_gnn, _ = self.cross_attention(
            gnn_features, transformer_features, transformer_features,
            key_padding_mask=~node_mask.bool() if node_mask is not None else None
        )
        
        enhanced_transformer, _ = self.cross_attention(
            transformer_features, gnn_features, gnn_features,
            key_padding_mask=~node_mask.bool() if node_mask is not None else None
        )
        
        gnn_weight = fusion_weights[..., 0:1]
        transformer_weight = fusion_weights[..., 1:2]
        
        fused_features = (gnn_weight * enhanced_gnn +
                         transformer_weight * enhanced_transformer)
        
        enhanced_features = self.feature_enhance(fused_features)
        
        output = self.layer_norm(fused_features + enhanced_features)
        
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)
        
        return output, fusion_weights


class GNNTransformerHybrid(nn.Module):
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 output_dim: int = 512,
                 gnn_layers: int = 3,
                 transformer_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 use_dynamic_fusion: bool = True,
                 use_gnn: bool = True,
                 use_transformer: bool = True,
                 gnn_backbone: str = 'graph_attention',
                 gcn_stack_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_dynamic_fusion = use_dynamic_fusion
        self.use_gnn = use_gnn
        self.use_transformer = use_transformer
        self.gnn_backbone = gnn_backbone

        # Attention visualization support
        self.store_attention = False
        self.attention_weights_cache = None  # [batch, heads, atoms, atoms]

        # GNN branch (optional based on ablation)
        self.gnn_branch = None
        if use_gnn:
            if gnn_backbone == 'pyg_gcn_stack':
                from .gcn_stack import GCNStack
                self.gnn_branch = GCNStack(
                    in_channels=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=gcn_stack_layers,
                    dropout=dropout,
                    use_residual=True
                )
            else:  # default: graph_attention
                self.gnn_branch = GraphAttentionNetwork(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=gnn_layers,
                    dropout=dropout
                )

        # Transformer branch (optional based on ablation)
        self.transformer_branch = None
        if use_transformer:
            self.transformer_branch = MolecularTransformer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=transformer_layers,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=max_seq_len
            )

        # Fusion module (only needed when both branches are active)
        if use_gnn and use_transformer:
            if use_dynamic_fusion:
                self.fusion_module = DynamicFusionModule(
                    feature_dim=hidden_dim,
                    num_heads=num_heads
                )
            else:
                self.fusion_module = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim)
                )
        else:
            self.fusion_module = None

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.num_heads = num_heads
        self.global_pooling = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x, edge_index, batch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dense_x, node_mask = to_dense_batch(x, batch)
        fusion_weights = None

        # Compute features based on which branches are active
        gnn_features = None
        transformer_features = None

        if self.use_gnn and self.gnn_branch is not None:
            if self.gnn_backbone == 'pyg_gcn_stack':
                # GCNStack returns [num_nodes, hidden_dim], need to convert to dense batch
                gnn_node_features = self.gnn_branch(x, edge_index)
                gnn_features, _ = to_dense_batch(gnn_node_features, batch)
            else:
                gnn_features = self.gnn_branch(x, edge_index, batch)

        if self.use_transformer and self.transformer_branch is not None:
            transformer_features = self.transformer_branch(
                node_features=dense_x,
                attention_mask=node_mask
            )

        # Determine fused features based on active branches
        if self.use_gnn and self.use_transformer and self.fusion_module is not None:
            # Both branches active - use fusion
            if self.use_dynamic_fusion:
                fused_features, fusion_weights = self.fusion_module(
                    gnn_features, transformer_features, node_mask
                )
            else:
                combined = torch.cat([gnn_features, transformer_features], dim=-1)
                fused_features = self.fusion_module(combined)
                if node_mask is not None:
                    fused_features = fused_features * node_mask.unsqueeze(-1)
        elif self.use_gnn and gnn_features is not None:
            # GNN only
            fused_features = gnn_features
        elif self.use_transformer and transformer_features is not None:
            # Transformer only
            fused_features = transformer_features
        else:
            # Fallback: use input directly
            fused_features = dense_x

        node_representations = self.output_layer(fused_features)

        global_repr, attention_weights = self.global_pooling(
            node_representations, node_representations, node_representations,
            key_padding_mask=~node_mask.bool() if node_mask is not None else None
        )

        # Store attention weights for visualization
        if self.store_attention and attention_weights is not None:
            self.attention_weights_cache = attention_weights.detach().clone()

        if node_mask is not None:
            mask_expanded = node_mask.unsqueeze(-1).float()
            masked_repr = global_repr * mask_expanded
            node_counts = node_mask.sum(dim=1, keepdim=True).float()
            molecule_repr = masked_repr.sum(dim=1) / (node_counts + 1e-8)
        else:
            molecule_repr = global_repr.mean(dim=1)

        interp_data = {
            'attention_weights': attention_weights,
            'fusion_weights': fusion_weights
        }

        return molecule_repr, interp_data


if __name__ == "__main__":
    hybrid_model = GNNTransformerHybrid(
        input_dim=256,
        hidden_dim=512,
        output_dim=512,
        gnn_layers=3,
        transformer_layers=6,
        num_heads=8,
        dropout=0.1,
        use_dynamic_fusion=True
    )
    
    batch_size = 4
    num_nodes = 64
    input_dim = 256
    
    node_features = torch.randn(batch_size, num_nodes, input_dim)
    node_mask = torch.ones(batch_size, num_nodes)
    
    try:
        molecule_repr, attention_weights = hybrid_model(
            node_features=node_features,
            node_mask=node_mask
        )
        
        print(f"Molecule representation shape: {molecule_repr.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")
        print("GNN-Transformer hybrid architecture created successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()