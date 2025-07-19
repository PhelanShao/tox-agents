"""
几何-拓扑双重编码器 (Geometric-Topological Dual Encoder)
同时利用分子的3D几何信息和2D图拓扑结构进行分子表示学习

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import math


class SE3EquivariantLayer(nn.Module):
    """SE(3)等变层 - 处理3D几何信息的等变神经网络层"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  
        self.output_dim = output_dim
        
        # 标量特征处理
        self.scalar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 向量特征处理
        self.vector_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 几何注意力权重
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),  # +3 for distance features
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def compute_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """计算原子间距离矩阵"""
        # coords: [batch_size, num_atoms, 3]
        batch_size, num_atoms, _ = coords.shape
        
        # 扩展坐标用于计算距离
        coords_i = coords.unsqueeze(2)  # [batch_size, num_atoms, 1, 3]
        coords_j = coords.unsqueeze(1)  # [batch_size, 1, num_atoms, 3]
        
        # 计算距离
        diff = coords_i - coords_j  # [batch_size, num_atoms, num_atoms, 3]
        distances = torch.norm(diff, dim=-1, p=2)  # [batch_size, num_atoms, num_atoms]
        
        return distances, diff
    
    def forward(self, 
                node_features: torch.Tensor,
                coordinates: torch.Tensor,
                edge_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SE(3)等变前向传播
        
        Args:
            node_features: 节点特征 [batch_size, num_atoms, input_dim]
            coordinates: 3D坐标 [batch_size, num_atoms, 3]
            edge_mask: 边掩码 [batch_size, num_atoms, num_atoms]
            
        Returns:
            scalar_out: 标量输出 [batch_size, num_atoms, output_dim]
            vector_out: 向量输出 [batch_size, num_atoms, 3]
        """
        batch_size, num_atoms, _ = node_features.shape
        
        # 计算距离和方向向量
        distances, directions = self.compute_distances(coordinates)
        
        # 标量特征处理
        scalar_features = self.scalar_net(node_features)
        
        # 几何注意力计算
        dist_features = torch.stack([
            distances,
            1.0 / (distances + 1e-6),  # 倒数距离
            torch.exp(-distances)      # 指数衰减距离
        ], dim=-1)  # [batch_size, num_atoms, num_atoms, 3]
        
        # 为每个原子对计算注意力权重
        attention_input = torch.cat([
            node_features.unsqueeze(2).expand(-1, -1, num_atoms, -1),
            dist_features
        ], dim=-1)
        
        attention_weights = self.attention_net(attention_input).squeeze(-1)
        attention_weights = attention_weights.masked_fill(~edge_mask.bool(), float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # 向量特征聚合 (SE(3)等变)
        vector_messages = directions * attention_weights.unsqueeze(-1)
        vector_aggregated = vector_messages.sum(dim=2)  # [batch_size, num_atoms, 3]
        
        # 向量特征变换
        vector_norms = torch.norm(vector_aggregated, dim=-1, keepdim=True)
        vector_features = self.vector_net(node_features)  # [batch_size, num_atoms, output_dim]
        
        # 向量输出 (保持方向，调整幅度)
        vector_out = vector_aggregated * vector_features[..., :3]  # 取前3维作为向量权重
        
        return scalar_features, vector_out


class GraphAttentionLayer(nn.Module):
    """图注意力层 - 处理2D拓扑结构"""
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0
        
        # 线性变换层
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim) 
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.out_linear = nn.Linear(output_dim, output_dim)
        
        # 边特征编码
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, output_dim),  # 键类型、距离等特征
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self,
                node_features: torch.Tensor,
                adjacency: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        图注意力前向传播
        
        Args:
            node_features: 节点特征 [batch_size, num_atoms, input_dim]
            adjacency: 邻接矩阵 [batch_size, num_atoms, num_atoms]
            edge_features: 边特征 [batch_size, num_atoms, num_atoms, edge_dim]
            
        Returns:
            输出节点特征 [batch_size, num_atoms, output_dim]
        """
        batch_size, num_atoms, _ = node_features.shape
        
        # 计算Q, K, V
        Q = self.q_linear(node_features)  # [batch_size, num_atoms, output_dim]
        K = self.k_linear(node_features)
        V = self.v_linear(node_features)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        scores = torch.einsum('bihd,bjhd->bhij', Q, K) / math.sqrt(self.head_dim)
        
        # 边特征增强
        if edge_features is not None:
            edge_embeds = self.edge_encoder(edge_features)  # [batch_size, num_atoms, num_atoms, output_dim]
            edge_embeds = edge_embeds.view(batch_size, num_atoms, num_atoms, self.num_heads, self.head_dim)
            edge_bias = torch.einsum('bihd,bijhd->bhij', Q, edge_embeds)
            scores = scores + edge_bias
        
        # 应用邻接矩阵掩码
        adjacency_mask = adjacency.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(~adjacency_mask.bool(), float('-inf'))
        
        # 注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 聚合特征
        out = torch.einsum('bhij,bjhd->bihd', attention_weights, V)
        out = out.contiguous().view(batch_size, num_atoms, self.output_dim)
        
        # 残差连接和层归一化
        if self.input_dim == self.output_dim:
            out = out + node_features
        out = self.norm(out)
        
        return out


class GeometricTopologicalEncoder(nn.Module):
    """
    几何-拓扑双重编码器主类
    同时处理3D几何信息和2D图拓扑结构
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 output_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 输入特征编码
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        
        # SE(3)等变几何编码器层
        self.geometric_layers = nn.ModuleList([
            SE3EquivariantLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 图注意力拓扑编码器层
        self.topological_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 几何-拓扑信息融合模块
        self.geo_topo_fusion = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 全局池化注意力
        self.global_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def create_edge_features(self, 
                           coordinates: torch.Tensor, 
                           bond_types: torch.Tensor,
                           distances: torch.Tensor) -> torch.Tensor:
        """创建边特征"""
        batch_size, num_atoms, _ = coordinates.shape
        
        # 基本边特征: [键类型, 距离, 角度, 二面角]
        edge_features = torch.zeros(batch_size, num_atoms, num_atoms, 4, 
                                  device=coordinates.device)
        
        # 键类型特征
        edge_features[..., 0] = bond_types
        
        # 距离特征 (归一化)
        edge_features[..., 1] = distances / (distances.max() + 1e-6)
        
        # 简化的角度特征 (实际实现中可以更复杂)
        edge_features[..., 2] = torch.cos(distances * math.pi)
        edge_features[..., 3] = torch.sin(distances * math.pi)
        
        return edge_features
    
    def forward(self,
                node_features: torch.Tensor,
                coordinates: torch.Tensor,
                adjacency: torch.Tensor,
                bond_types: torch.Tensor,
                atom_mask: torch.Tensor) -> torch.Tensor:
        """
        几何-拓扑双重编码前向传播
        
        Args:
            node_features: 节点特征 [batch_size, num_atoms, input_dim]
            coordinates: 3D坐标 [batch_size, num_atoms, 3]
            adjacency: 邻接矩阵 [batch_size, num_atoms, num_atoms]
            bond_types: 键类型 [batch_size, num_atoms, num_atoms]
            atom_mask: 原子掩码 [batch_size, num_atoms]
            
        Returns:
            融合表示 [batch_size, output_dim]
        """
        batch_size, num_atoms, _ = node_features.shape
        
        # 输入编码
        h = self.input_encoder(node_features)  # [batch_size, num_atoms, hidden_dim]
        
        # 计算距离矩阵
        coords_i = coordinates.unsqueeze(2)
        coords_j = coordinates.unsqueeze(1)
        distances = torch.norm(coords_i - coords_j, dim=-1, p=2)
        
        # 创建边特征
        edge_features = self.create_edge_features(coordinates, bond_types, distances)
        
        # 逐层处理
        for i in range(self.num_layers):
            # 几何编码 (SE(3)等变)
            geometric_scalar, geometric_vector = self.geometric_layers[i](
                h, coordinates, adjacency
            )
            
            # 拓扑编码 (图注意力)
            topological_features = self.topological_layers[i](
                h, adjacency, edge_features
            )
            
            # 几何-拓扑融合
            # 将几何向量信息融入标量特征
            vector_magnitude = torch.norm(geometric_vector, dim=-1, keepdim=True)
            enhanced_geometric = torch.cat([geometric_scalar, vector_magnitude], dim=-1)
            
            # 调整维度以匹配
            if enhanced_geometric.shape[-1] != topological_features.shape[-1]:
                enhanced_geometric = enhanced_geometric[..., :topological_features.shape[-1]]
            
            # 交叉注意力融合
            fused_features, _ = self.geo_topo_fusion[i](
                enhanced_geometric, topological_features, topological_features
            )
            
            # 残差连接和层归一化
            h = h + fused_features
            h = self.layer_norms[i](h)
        
        # 输出投影
        h = self.output_projection(h)  # [batch_size, num_atoms, output_dim]
        
        # 全局表示聚合
        # 使用自注意力进行全局池化
        global_repr, attention_weights = self.global_attention(h, h, h)
        
        # 考虑原子掩码的平均池化
        mask_expanded = atom_mask.unsqueeze(-1).float()
        masked_repr = global_repr * mask_expanded
        atom_counts = atom_mask.sum(dim=1, keepdim=True).float()
        final_repr = masked_repr.sum(dim=1) / (atom_counts + 1e-8)
        
        return final_repr


class GeometricTopologicalFusion(nn.Module):
    """几何-拓扑信息融合模块"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 几何信息编码
        self.geo_encoder = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),  # +3 for vector features
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 拓扑信息编码
        self.topo_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, 
                geometric_features: torch.Tensor,
                geometric_vectors: torch.Tensor,
                topological_features: torch.Tensor) -> torch.Tensor:
        """
        几何-拓扑信息融合
        
        Args:
            geometric_features: 几何标量特征 [batch_size, num_atoms, feature_dim]
            geometric_vectors: 几何向量特征 [batch_size, num_atoms, 3]
            topological_features: 拓扑特征 [batch_size, num_atoms, feature_dim]
            
        Returns:
            融合特征 [batch_size, num_atoms, feature_dim]
        """
        # 几何信息编码 (结合标量和向量)
        geo_input = torch.cat([geometric_features, geometric_vectors], dim=-1)
        geo_encoded = self.geo_encoder(geo_input)
        
        # 拓扑信息编码
        topo_encoded = self.topo_encoder(topological_features)
        
        # 交叉注意力融合
        fused_geo, _ = self.cross_attention(geo_encoded, topo_encoded, topo_encoded)
        fused_topo, _ = self.cross_attention(topo_encoded, geo_encoded, geo_encoded)
        
        # 门控融合
        combined = torch.cat([fused_geo, fused_topo], dim=-1)
        gate_weights = self.gate(combined)
        
        final_features = gate_weights * fused_geo + (1 - gate_weights) * fused_topo
        
        return final_features


# 使用示例
if __name__ == "__main__":
    # 创建几何-拓扑双重编码器
    encoder = GeometricTopologicalEncoder(
        input_dim=256,
        hidden_dim=512,
        output_dim=512,
        num_layers=4
    )
    
    # 模拟输入数据
    batch_size = 4
    num_atoms = 64
    
    node_features = torch.randn(batch_size, num_atoms, 256)
    coordinates = torch.randn(batch_size, num_atoms, 3)
    adjacency = torch.randint(0, 2, (batch_size, num_atoms, num_atoms)).float()
    bond_types = torch.randint(0, 4, (batch_size, num_atoms, num_atoms)).float()
    atom_mask = torch.ones(batch_size, num_atoms)
    
    # 前向传播
    try:
        output = encoder(node_features, coordinates, adjacency, bond_types, atom_mask)
        print(f"输出形状: {output.shape}")
        print(f"几何-拓扑双重编码器创建成功!")
        
        # 输出一些统计信息
        print(f"输出均值: {output.mean().item():.4f}")
        print(f"输出标准差: {output.std().item():.4f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 