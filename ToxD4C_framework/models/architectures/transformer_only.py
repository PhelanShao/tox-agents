"""
Transformer-only architecture for ablation studies.

Used when the GNN branch is disabled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
import math


class PositionalEncoding(nn.Module):
    """Positional encoding module."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class TransformerOnly(nn.Module):
    """Transformer-only encoder used in ablation studies."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_atoms: int = 200):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_atoms = max_atoms
        
        # Input projection
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_atoms)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False  # [seq_len, batch, features]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: node features with shape [num_nodes, input_dim]
            edge_index: edge indices [2, num_edges] (ignored in this encoder)
            batch: batch indices for each node [num_nodes]

        Returns:
            graph_representation: pooled graph embedding with shape [batch_size, hidden_dim]
        """
        batch_size = batch.max().item() + 1 if batch is not None else 1

        # Project to the hidden dimension
        x = self.input_projection(x)  # [num_nodes, hidden_dim]

        # Convert to a dense representation per batch element
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_atoms)
        # x_dense: [batch_size, max_atoms, hidden_dim]
        # mask: [batch_size, max_atoms]

        # Reshape to [seq_len, batch, features] for the transformer
        x_dense = x_dense.transpose(0, 1)  # [max_atoms, batch_size, hidden_dim]

        # Add positional encodings
        x_dense = self.pos_encoding(x_dense)

        # Create the attention mask (True denotes padding locations)
        attn_mask = ~mask

        # Transformer encoder pass (expects [batch_size, seq_len] padding mask)
        encoded = self.transformer_encoder(
            x_dense, 
            src_key_padding_mask=attn_mask
        )  # [max_atoms, batch_size, hidden_dim]

        # Convert back to [batch_size, max_atoms, hidden_dim]
        encoded = encoded.transpose(0, 1)

        # Apply the mask followed by mean pooling
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
        encoded_masked = encoded * mask_expanded.float()

        # Compute a masked average for each graph
        graph_lengths = mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
        graph_representation = encoded_masked.sum(dim=1) / graph_lengths.clamp(min=1)
        # [batch_size, hidden_dim]

        # Output projection and dropout
        graph_representation = self.output_projection(graph_representation)
        graph_representation = self.dropout(graph_representation)

        return graph_representation

    def get_attention_weights(self, x, edge_index, batch=None):
        """Hook for extracting attention weights for visualisation."""
        # The implementation can be added when attention visualisation is required.
        return None


class TransformerOnlyWithEdges(TransformerOnly):
    """Transformer-only encoder that incorporates edge-aware attention biases."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Optional edge embedding used to bias attention scores
        self.edge_embedding = nn.Linear(1, self.num_heads)

    def create_attention_bias(self, edge_index, batch, max_atoms):
        """
        Create an attention bias tensor from the graph connectivity.

        Args:
            edge_index: edge indices [2, num_edges]
            batch: batch indices for each node [num_nodes]
            max_atoms: maximum number of atoms per graph

        Returns:
            attention_bias: tensor with shape [batch_size, num_heads, max_atoms, max_atoms]
        """
        batch_size = batch.max().item() + 1
        device = edge_index.device

        # Initialise bias tensor
        attention_bias = torch.zeros(
            batch_size, self.num_heads, max_atoms, max_atoms,
            device=device
        )

        # Build adjacency information per batch example
        for b in range(batch_size):
            # Select the nodes belonging to this batch element
            node_mask = (batch == b)
            node_indices = torch.where(node_mask)[0]

            if len(node_indices) == 0:
                continue

            # Map global indices to local indices
            global_to_local = {global_idx.item(): local_idx 
                             for local_idx, global_idx in enumerate(node_indices)}

            # Identify edges inside the current batch element
            edge_mask = torch.isin(edge_index[0], node_indices) & \
                       torch.isin(edge_index[1], node_indices)
            batch_edges = edge_index[:, edge_mask]

            # Convert to local indices
            for edge_idx in range(batch_edges.size(1)):
                src_global = batch_edges[0, edge_idx].item()
                dst_global = batch_edges[1, edge_idx].item()
                
                if src_global in global_to_local and dst_global in global_to_local:
                    src_local = global_to_local[src_global]
                    dst_local = global_to_local[dst_global]
                    
                    # Encourage higher attention between connected nodes
                    attention_bias[b, :, src_local, dst_local] = 1.0
                    attention_bias[b, :, dst_local, src_local] = 1.0  # undirected graph
        
        return attention_bias
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass with optional edge-aware adjustments."""
        # Delegate to the base transformer encoder for now
        return super().forward(x, edge_index, batch)
        # TODO: incorporate attention_bias derived from edge_index when required
