"""
GCN Stack Module - PyG GCNConv based node encoder

This module provides a residual GCN stack using PyTorch Geometric's GCNConv layers.
It can be used as an alternative GNN backbone in the ToxD4C model.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNStack(nn.Module):
    """
    GCN Stack encoder based on PyG GCNConv (2-4 layers recommended), with residual connections and LayerNorm.
    
    Args:
        in_channels: Input feature dimension
        hidden_dim: Hidden layer dimension  
        num_layers: Number of GCN layers (recommended 2-4)
        dropout: Dropout rate
        use_residual: Whether to use residual connections
        
    Input:
        x: Node features [num_nodes, in_channels]
        edge_index: Graph connectivity [2, num_edges]
        
    Output:
        Node features [num_nodes, hidden_dim]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        assert num_layers >= 2, "GCNStack: recommend 2-4 layers"
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: in_channels -> hidden_dim
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Projection for residual if first layer has different dimensions
        self.res_proj = None
        if use_residual and in_channels != hidden_dim:
            self.res_proj = nn.Linear(in_channels, hidden_dim, bias=False)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GCN stack.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)

            if self.use_residual:
                # First layer residual: need dimension alignment
                if i == 0 and self.res_proj is not None:
                    h = h + self.res_proj(h_in)
                elif i > 0:
                    h = h + h_in

        return h


if __name__ == "__main__":
    # Test the GCN Stack
    import torch
    
    # Create test data
    num_nodes = 20
    in_channels = 512
    hidden_dim = 512
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.tensor([
        list(range(num_nodes)) + list(range(1, num_nodes)) + [0],
        list(range(1, num_nodes)) + [0] + list(range(num_nodes))
    ], dtype=torch.long)
    
    # Test model
    model = GCNStack(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1,
        use_residual=True
    )
    
    output = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print("GCNStack test passed!")

