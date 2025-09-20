import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNStack(nn.Module):
    """
    基于 PyG GCNConv 的节点编码器（2-4层可配），带残差与LayerNorm。
    - 输入: 节点特征 [num_nodes, in_channels]
    - 输出: 节点特征 [num_nodes, hidden_dim]
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
        assert num_layers >= 2, "GCNStack: 建议使用 2-4 层"
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 第一层: in_channels -> hidden_dim
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # 中间层: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # 若首层维度不同，为残差准备投影
        self.res_proj = None
        if use_residual and in_channels != hidden_dim:
            self.res_proj = nn.Linear(in_channels, hidden_dim, bias=False)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_in = h
            h = conv(h, edge_index)
            h = norm(h)
            h = self.activation(h)
            h = self.dropout(h)

            if self.use_residual:
                # 第一层 residual: 需要维度对齐
                if i == 0 and self.res_proj is not None:
                    h = h + self.res_proj(h_in)
                elif i > 0:
                    h = h + h_in

        return h

