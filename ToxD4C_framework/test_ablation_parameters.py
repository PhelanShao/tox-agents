#!/usr/bin/env python3
"""
测试消融实验参数传递和模型构建
验证每个消融实验的模型参数量是否符合预期
"""

import sys
import torch
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from models.toxd4c import ToxD4C
from configs.toxd4c_config import get_enhanced_toxd4c_config


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """计算模型大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def test_ablation_config(config_name, config_modifications):
    """测试特定消融配置"""
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"{'='*60}")
    
    # 获取基础配置
    config = get_enhanced_toxd4c_config()
    
    # 应用修改
    for key, value in config_modifications.items():
        config[key] = value
        print(f"  {key}: {value}")
    
    try:
        # 创建模型
        device = 'cpu'  # 使用CPU进行测试
        model = ToxD4C(config, device=device)
        
        # 计算参数量
        total_params, trainable_params = count_parameters(model)
        model_size_mb = get_model_size_mb(model)
        
        print(f"\n模型统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        print(f"  模型大小: {model_size_mb:.2f} MB")
        
        # 检查模型组件
        print(f"\n模型组件:")
        print(f"  主编码器类型: {type(model.main_encoder).__name__}")
        print(f"  几何编码器: {'✓' if hasattr(model, 'geometric_encoder') else '✗'}")
        print(f"  层次编码器: {'✓' if hasattr(model, 'hierarchical_encoder') else '✗'}")
        print(f"  指纹模块: {'✓' if hasattr(model, 'fingerprint_module') else '✗'}")
        print(f"  对比学习: {'✓' if hasattr(model, 'contrastive_loss') else '✗'}")
        
        # 测试前向传播
        print(f"\n前向传播测试:")
        test_forward_pass(model, device)
        
        return {
            'config_name': config_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'success': True,
            'encoder_type': type(model.main_encoder).__name__
        }
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'config_name': config_name,
            'success': False,
            'error': str(e)
        }


def test_forward_pass(model, device):
    """测试模型前向传播"""
    try:
        # 创建测试数据
        batch_size = 2
        num_atoms = 10
        
        test_data = {
            'atom_features': torch.randn(batch_size * num_atoms, 119),  # 使用正确的原子特征维度
            'edge_index': torch.randint(0, batch_size * num_atoms, (2, 20)),
            'batch': torch.repeat_interleave(torch.arange(batch_size), num_atoms),
            'coordinates': torch.randn(batch_size * num_atoms, 3)
        }
        
        test_smiles = ['CCO', 'CCN']
        
        model.eval()
        with torch.no_grad():
            output = model(test_data, test_smiles)
            
        print(f"    分类输出形状: {output['predictions']['classification'].shape}")
        print(f"    回归输出形状: {output['predictions']['regression'].shape}")
        print(f"    图表示形状: {output['graph_representation'].shape}")
        print(f"    ✓ 前向传播成功")
        
    except Exception as e:
        print(f"    ❌ 前向传播失败: {str(e)}")
        raise


def main():
    """运行所有消融实验测试"""
    print("🧪 消融实验参数传递验证")
    print("="*80)
    
    # 定义所有消融实验配置
    ablation_configs = {
        "完整模型": {},
        
        "GNN Only": {
            'use_transformer': False,
            'use_geometric_encoder': False,
            'use_hierarchical_encoder': False,
            'use_fingerprints': False,
            'use_hybrid_architecture': False
        },
        
        "GNN + Transformer": {
            'use_geometric_encoder': False,
            'use_hierarchical_encoder': False,
            'use_fingerprints': False
        },
        
        "GNN + Trans + 3D": {
            'use_hierarchical_encoder': False,
            'use_fingerprints': False
        },
        
        "GNN + Trans + FP": {
            'use_geometric_encoder': False,
            'use_hierarchical_encoder': False
        },
        
        "Full - Contrastive": {
            'use_contrastive_learning': False
        },
        
        "Full - Fingerprint": {
            'use_fingerprints': False
        },
        
        "Concatenation Fusion": {
            'use_dynamic_fusion': False,
            'fusion_method': 'concatenation'
        },
        
        "Classification Only": {
            'enable_regression': False
        },
        
        "Regression Only": {
            'enable_classification': False
        },
        
        "Transformer Only": {
            'use_gnn': False,
            'use_geometric_encoder': False,
            'use_hierarchical_encoder': False,
            'use_fingerprints': False,
            'use_hybrid_architecture': False
        },
        
        "No GNN No Transformer": {
            'use_gnn': False,
            'use_transformer': False,
            'use_geometric_encoder': False,
            'use_hierarchical_encoder': False,
            'use_fingerprints': False,
            'use_hybrid_architecture': False
        }
    }
    
    # 运行所有测试
    results = []
    for config_name, modifications in ablation_configs.items():
        result = test_ablation_config(config_name, modifications)
        results.append(result)
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("消融实验参数量汇总")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if successful_results:
        print(f"{'配置名称':<20} {'参数量':<12} {'大小(MB)':<10} {'编码器类型':<20}")
        print("-" * 70)
        
        for result in successful_results:
            print(f"{result['config_name']:<20} "
                  f"{result['total_params']:>10,} "
                  f"{result['model_size_mb']:>8.1f} "
                  f"{result['encoder_type']:<20}")
    
    # 检查参数量变化是否合理
    print(f"\n{'='*80}")
    print("参数量变化分析")
    print(f"{'='*80}")
    
    if len(successful_results) >= 2:
        full_model = next((r for r in successful_results if r['config_name'] == '完整模型'), None)
        gnn_only = next((r for r in successful_results if r['config_name'] == 'GNN Only'), None)
        
        if full_model and gnn_only:
            reduction = full_model['total_params'] - gnn_only['total_params']
            reduction_pct = (reduction / full_model['total_params']) * 100
            
            print(f"完整模型参数量: {full_model['total_params']:,}")
            print(f"GNN Only参数量: {gnn_only['total_params']:,}")
            print(f"参数量减少: {reduction:,} ({reduction_pct:.1f}%)")
            
            if reduction_pct > 50:
                print("✅ 消融实验参数量减少合理")
            else:
                print("⚠️  消融实验参数量减少不明显，可能存在问题")
    
    # 失败的配置
    failed_results = [r for r in results if not r.get('success', False)]
    if failed_results:
        print(f"\n❌ 失败的配置:")
        for result in failed_results:
            print(f"  {result['config_name']}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
