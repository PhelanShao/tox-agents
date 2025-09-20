import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from torch_geometric.nn import GCN, global_mean_pool
from .architectures.gcn_stack import GCNStack
from torch_geometric.utils import to_dense_batch

from .architectures.gnn_transformer_hybrid import GNNTransformerHybrid
from .encoders.geometric_encoder import GeometricEncoder
from .encoders.hierarchical_encoder import HierarchicalEncoder
from .fingerprints.molecular_fingerprint_enhanced import MolecularFingerprintModule
from .heads.multi_scale_prediction_head import MultiScalePredictionHead
from configs.toxd4c_config import CLASSIFICATION_TASKS, REGRESSION_TASKS
from .losses.contrastive_loss import SupConLoss


class ToxD4C(nn.Module):
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__()
        
        self.config = config
        self.device = device
        
        self.atom_embedding = nn.Linear(
            config['atom_features_dim'], config['hidden_dim']
        )

        # 消融实验：检查是否禁用GNN或Transformer
        use_gnn = config.get('use_gnn', True)
        use_transformer = config.get('use_transformer', True)

        if not use_gnn and not use_transformer:
            # 如果两者都禁用，使用简单的MLP
            self.main_encoder = nn.Sequential(
                nn.Linear(config['hidden_dim'], config['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(config['dropout']),
                nn.Linear(config['hidden_dim'], config['hidden_dim'])
            )
        elif use_gnn and use_transformer and config.get('use_hybrid_architecture', False):
            # 使用混合架构
            self.main_encoder = GNNTransformerHybrid(
                input_dim=config['hidden_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['hidden_dim'],
                gnn_layers=config.get('gnn_layers', 3),
                transformer_layers=config.get('transformer_layers', 3),
                num_heads=config['num_attention_heads'],
                dropout=config['dropout'],
                use_dynamic_fusion=config.get('use_dynamic_fusion', True),
                gnn_backbone=config.get('gnn_backbone', 'graph_attention')
            )
        elif use_gnn and not use_transformer:
            # 仅使用GNN
            if config.get('gnn_backbone', 'graph_attention') == 'pyg_gcn_stack':
                self.main_encoder = GCNStack(
                    in_channels=config['hidden_dim'],
                    hidden_dim=config['hidden_dim'],
                    num_layers=config.get('gcn_stack_layers', 3),
                    dropout=config['dropout'],
                    use_residual=True,
                )
            else:
                self.main_encoder = GCN(
                    in_channels=config['hidden_dim'],
                    hidden_channels=config['hidden_dim'],
                    num_layers=config['num_encoder_layers'],
                    dropout=config['dropout']
                )
        elif not use_gnn and use_transformer:
            # 仅使用Transformer (需要实现)
            from .architectures.transformer_only import TransformerOnly
            self.main_encoder = TransformerOnly(
                input_dim=config['hidden_dim'],
                hidden_dim=config['hidden_dim'],
                num_layers=config.get('transformer_layers', 3),
                num_heads=config['num_attention_heads'],
                dropout=config['dropout']
            )
        else:
            # 默认使用GCN
            self.main_encoder = GCN(
                in_channels=config['hidden_dim'],
                hidden_channels=config['hidden_dim'],
                num_layers=config['num_encoder_layers'],
                dropout=config['dropout']
            )

        if config.get('use_geometric_encoder', False):
            self.geometric_encoder = GeometricEncoder(
                embed_dim=config['hidden_dim'],
                hidden_dim=config.get('geometric_hidden_dim', 256),
                num_rbf=config.get('num_rbf', 50),
                max_distance=config.get('max_distance', 10.0)
            )

        if config.get('use_hierarchical_encoder', False):
            self.hierarchical_encoder = HierarchicalEncoder(
                input_dim=config['hidden_dim'],
                hidden_dim=config['hidden_dim'],
                hierarchy_levels=config.get('hierarchy_levels', [2, 4, 8]),
                dropout=config['dropout']
            )
            
        if config.get('use_fingerprints', False):
            self.fingerprint_module = MolecularFingerprintModule(
                output_dim=config.get('fingerprint_dim', 512),
                fingerprint_configs=config.get('fingerprint_configs', {})
            )
            fp_dim = config.get('fingerprint_dim', 512)
        else:
            fp_dim = 0

        fusion_input_dim = config['hidden_dim']
        if config.get('use_hierarchical_encoder', False):
            fusion_input_dim += config['hidden_dim']
        if config.get('use_fingerprints', False):
            fusion_input_dim += fp_dim
            
        self.fusion_layer = nn.Linear(fusion_input_dim, config['hidden_dim'])
        
        # 根据配置启用/禁用任务分支（用于消融）
        classification_enabled = config.get('enable_classification', True)
        regression_enabled = config.get('enable_regression', True)

        enabled_cls_tasks = CLASSIFICATION_TASKS if classification_enabled else []
        enabled_reg_tasks = REGRESSION_TASKS if regression_enabled else []

        self.prediction_head = MultiScalePredictionHead(
            input_dim=config['hidden_dim'],
            task_configs=config['task_configs'],
            dropout=config['dropout'],
            uncertainty_weighting=config.get('uncertainty_weighting', False),
            classification_tasks_list=enabled_cls_tasks,
            regression_tasks_list=enabled_reg_tasks,
            single_endpoint_cls=config.get('single_endpoint_cls'),
            single_endpoint_reg=config.get('single_endpoint_reg')
        )
        
        if config.get('use_contrastive_learning', False):
            self.contrastive_loss = SupConLoss(
                temperature=config.get('contrastive_temperature', 0.1)
            )

    def forward(self, 
                data: Dict[str, torch.Tensor], 
                smiles_list: List[str]) -> Dict[str, Any]:
        x = data['atom_features']
        edge_index = data['edge_index']
        batch = data.get('batch')
        pos = data.get('coordinates')
        
        atom_repr = self.atom_embedding(x)
        
        if hasattr(self, 'geometric_encoder') and pos is not None:
            atom_repr = self.geometric_encoder(atom_repr, pos, edge_index)
        
        # 根据编码器类型调用不同的方法
        if isinstance(self.main_encoder, GCN):
            node_repr = self.main_encoder(atom_repr, edge_index)
            graph_repr_main = global_mean_pool(node_repr, batch)
        elif hasattr(self.main_encoder, 'forward') and 'GNNTransformerHybrid' in str(type(self.main_encoder)):
            # 混合架构
            graph_repr_main, _ = self.main_encoder(atom_repr, edge_index, batch)
        elif hasattr(self.main_encoder, 'forward') and 'TransformerOnly' in str(type(self.main_encoder)):
            # 仅Transformer架构
            graph_repr_main = self.main_encoder(atom_repr, edge_index, batch)
        elif isinstance(self.main_encoder, nn.Sequential):
            # 简单MLP（当GNN和Transformer都禁用时）
            node_repr = self.main_encoder(atom_repr)
            graph_repr_main = global_mean_pool(node_repr, batch)
        else:
            # 默认处理
            try:
                graph_repr_main, _ = self.main_encoder(atom_repr, edge_index, batch)
            except:
                node_repr = self.main_encoder(atom_repr, edge_index)
                graph_repr_main = global_mean_pool(node_repr, batch)
        
        all_graph_repr = [graph_repr_main]
        
        if hasattr(self, 'hierarchical_encoder'):
            graph_repr_hierarchical = self.hierarchical_encoder(atom_repr, edge_index, batch)
            all_graph_repr.append(graph_repr_hierarchical)
            
        if hasattr(self, 'fingerprint_module'):
            fp_repr = self.fingerprint_module(smiles=smiles_list)
            all_graph_repr.append(fp_repr)
            
        if len(all_graph_repr) > 1:
            fused_repr = torch.cat(all_graph_repr, dim=1)
            final_graph_repr = self.fusion_layer(fused_repr)
        else:
            final_graph_repr = graph_repr_main
            
        cls_preds, reg_preds = self.prediction_head(final_graph_repr)
        
        output = {
            'predictions': {
                'classification': cls_preds,
                'regression': reg_preds
            },
            'graph_representation': final_graph_repr,
            'uncertainties': None
        }
        
            
        if hasattr(self, 'contrastive_loss'):
            output['contrastive_features'] = graph_repr_main
            
        return output

    def compute_contrastive_loss(self, 
                                 features: torch.Tensor, 
                                 labels: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'contrastive_loss'):
            # Expect features shape [batch, dim]; SupConLoss expects [batch, dim]
            # Labels are continuous multi-task targets used to define similarity.
            try:
                if features.ndim > 2:
                    features = features.view(features.size(0), -1)
                return self.contrastive_loss(features, labels)
            except Exception:
                # Fallback: return zero to avoid training breakage
                return torch.tensor(0.0, device=self.device)
        
        return torch.tensor(0.0, device=self.device)


if __name__ == '__main__':
    from configs.toxd4c_config import get_enhanced_toxd4c_config
    
    config = get_enhanced_toxd4c_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ToxD4C(config, device=device).to(device)
    
    print("ToxD4C Model Test:")
    print(f"Device: {device}")
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    batch_size = 4
    num_atoms_per_mol = 30
    
    data = {
        'atom_features': torch.randn(batch_size * num_atoms_per_mol, config['atom_features_dim']).to(device),
        'bond_features': torch.randn(batch_size * 50, config['bond_features_dim']).to(device),
        'edge_index': torch.randint(0, batch_size * num_atoms_per_mol, (2, 100)).to(device),
        'coordinates': torch.randn(batch_size * num_atoms_per_mol, 3).to(device),
        'batch': torch.repeat_interleave(torch.arange(batch_size), num_atoms_per_mol).to(device)
    }
    
    smiles_list = ["CCO"] * batch_size
    
    output = model(data, smiles_list)
    
    print("\nForward pass output:")
    print(f"Graph representation shape: {output['graph_representation'].shape}")
    
    # Check the number of prediction tasks based on what's available
    num_cls_tasks = output['predictions']['classification'].shape[1] if 'classification' in output['predictions'] and output['predictions']['classification'].numel() > 0 else 0
    num_reg_tasks = output['predictions']['regression'].shape[1] if 'regression' in output['predictions'] and output['predictions']['regression'].numel() > 0 else 0
    print(f"Number of prediction tasks: {num_cls_tasks + num_reg_tasks}")

    # Check output for a specific task if available
    if num_cls_tasks > 0:
        task_name = CLASSIFICATION_TASKS[0]
        print(f"Prediction shape for task '{task_name}': {output['predictions']['classification'][:, 0].shape}")
        assert output['predictions']['classification'].shape[0] == batch_size
    elif num_reg_tasks > 0:
        task_name = REGRESSION_TASKS[0]
        print(f"Prediction shape for task '{task_name}': {output['predictions']['regression'][:, 0].shape}")
        assert output['predictions']['regression'].shape[0] == batch_size

    assert output['graph_representation'].shape == (batch_size, config['hidden_dim'])
    
    print("\n✓ Model test passed!")
