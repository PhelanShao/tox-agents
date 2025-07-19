import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from typing import Optional, Tuple


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class GeometricEncoder(MessagePassing):
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 num_rbf: int,
                 max_distance: float):
        super(GeometricEncoder, self).__init__(aggr='add')
        
        self.embed_dim = embed_dim
        
        self.distance_expansion = GaussianSmearing(
            start=0.0, stop=max_distance, num_gaussians=num_rbf
        )
        
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * embed_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.filter_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
        for layer in self.update_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, 
                x: torch.Tensor, 
                pos: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        
        dist_emb = self.distance_expansion(dist)
        
        updated_x = self.propagate(edge_index, x=x, dist_emb=dist_emb)
        
        x = x + self.update_net(updated_x)
        return x

    def message(self, x_j: torch.Tensor, dist_emb: torch.Tensor) -> torch.Tensor:
        filter_weights = self.filter_net(dist_emb).view(
            -1, self.embed_dim, self.embed_dim
        )
        
        x_j_reshaped = x_j.unsqueeze(1)
        
        message = torch.bmm(x_j_reshaped, filter_weights)
        
        return message.squeeze(1)


if __name__ == '__main__':
    num_atoms = 50
    embed_dim = 128
    hidden_dim = 256
    num_rbf = 50
    max_distance = 10.0
    
    atom_features = torch.randn(num_atoms, embed_dim)
    positions = torch.randn(num_atoms, 3)
    edge_index = torch.randint(0, num_atoms, (2, 200))
    
    geo_encoder = GeometricEncoder(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_rbf=num_rbf,
        max_distance=max_distance
    )
    
    print("Geometric Encoder Test:")
    print(f"Input atom features shape: {atom_features.shape}")
    print(f"Input coordinates shape: {positions.shape}")
    
    output = geo_encoder(atom_features, positions, edge_index)
    
    print(f"Output atom features shape: {output.shape}")
    
    assert output.shape == (num_atoms, embed_dim)
    print("✓ Test passed!")