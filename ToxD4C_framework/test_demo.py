"""
ToxD4C 快速测试演示
测试31个毒性预测任务的基本功能

作者: AI助手
日期: 2024-06-11
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

try:
    from configs.toxd4c_config import (
        get_enhanced_toxd4c_config, CLASSIFICATION_TASKS, REGRESSION_TASKS
    )
    from data.lmdb_dataset import LMDBToxD4CDataset, MolecularFeatureExtractor
    print("✓ 配置和数据模块导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)


def test_config():
    """测试配置"""
    print("\n=== 测试配置 ===")
    
    config = get_enhanced_toxd4c_config()
    
    print(f"分类任务数量: {len(CLASSIFICATION_TASKS)}")
    print(f"回归任务数量: {len(REGRESSION_TASKS)}")
    print(f"总任务数量: {len(CLASSIFICATION_TASKS + REGRESSION_TASKS)}")
    
    print(f"\n前5个分类任务:")
    for i, task in enumerate(CLASSIFICATION_TASKS[:5], 1):
        print(f"  {i}. {task}")
    
    print(f"\n所有回归任务:")
    for i, task in enumerate(REGRESSION_TASKS, 1):
        print(f"  {i}. {task}")
    
    print(f"\n模型配置:")
    print(f"  隐藏维度: {config['hidden_dim']}")
    print(f"  注意力头数: {config['num_attention_heads']}")
    print(f"  编码器层数: {config['num_encoder_layers']}")


def test_data_loading():
    """测试数据加载"""
    print("\n=== 测试数据加载 ===")
    
    try:
        config = get_enhanced_toxd4c_config()
        config['batch_size'] = 4  # 小批次便于测试
        
        train_loader, val_loader, test_loader = create_dataloaders(None, config)
        
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        print(f"测试集批次数: {len(test_loader)}")
        
        # 测试一个批次
        for batch in train_loader:
            print(f"\n批次数据形状:")
            print(f"  原子特征: {batch['data']['atom_features'].shape}")
            print(f"  键特征: {batch['data']['bond_features'].shape}")
            print(f"  边索引: {batch['data']['edge_index'].shape}")
            print(f"  坐标: {batch['data']['coordinates'].shape}")
            print(f"  批次索引: {batch['data']['batch'].shape}")
            print(f"  SMILES数量: {len(batch['smiles'])}")
            
            # 检查标签
            print(f"\n标签检查:")
            sample_tasks = ['Carcinogenicity', 'Acute oral toxicity (LD50)']
            for task in sample_tasks:
                if task in batch['targets']:
                    labels = batch['targets'][task]
                    print(f"  {task}: {labels.shape}, 有效值: {torch.sum(~torch.isnan(labels))}")
            
            break
        
        print("✓ 数据加载测试成功")
        
    except Exception as e:
        print(f"✗ 数据加载测试失败: {e}")


def test_feature_extraction():
    """测试特征提取"""
    print("\n=== 测试特征提取 ===")
    
    try:
        from rdkit import Chem
        
        extractor = MolecularFeatureExtractor()
        
        # 测试分子
        test_smiles = ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
        test_names = ["乙醇", "阿司匹林", "咖啡因"]
        
        for smiles, name in zip(test_smiles, test_names):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  ✗ {name}: 无效SMILES")
                continue
            
            graph_data = extractor.mol_to_graph(mol)
            if graph_data is None:
                print(f"  ✗ {name}: 特征提取失败")
                continue
            
            print(f"  ✓ {name}:")
            print(f"    原子数: {graph_data['num_atoms']}")
            print(f"    键数: {graph_data['num_bonds']}")
            print(f"    原子特征维度: {graph_data['atom_features'].shape}")
            print(f"    键特征维度: {graph_data['bond_features'].shape}")
        
        print("✓ 特征提取测试成功")
        
    except Exception as e:
        print(f"✗ 特征提取测试失败: {e}")


def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    
    try:
        # 简化模型配置
        config = {
            'atom_features_dim': 119,
            'bond_features_dim': 12,
            'hidden_dim': 128,
            'num_encoder_layers': 2,
            'num_attention_heads': 4,
            'dropout': 0.1,
            'task_configs': {},
            'use_geometric_encoder': False,
            'use_hierarchical_encoder': False,
            'use_hybrid_architecture': False,
            'use_fingerprints': False,
            'use_contrastive_learning': False
        }
        
        # 添加任务配置
        for task in CLASSIFICATION_TASKS:
            config['task_configs'][task] = {
                'output_dim': 1,
                'task_type': 'classification'
            }
        
        for task in REGRESSION_TASKS:
            config['task_configs'][task] = {
                'output_dim': 1,
                'task_type': 'regression'
            }
        
        # 创建简化版本的模型
        from models.toxd4c import ToxD4C

        model = ToxD4C(config=config, device='cpu')
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ 模型创建成功")
        print(f"  总参数数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  支持任务数: {len(config['task_configs'])}")
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")


def test_forward_pass():
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")
    
    try:
        print("[FWD_PASS] 1. Importing modules...")
        from rdkit import Chem
        
        print("[FWD_PASS] 2. Creating simplified config...")
        config = {
            'atom_features_dim': 119,
            'bond_features_dim': 12,
            'hidden_dim': 64,
            'num_encoder_layers': 1,
            'num_attention_heads': 2,
            'dropout': 0.1,
            'task_configs': {},
            'use_geometric_encoder': False,
            'use_hierarchical_encoder': False,
            'use_hybrid_architecture': False, # This forces the use of the GCN encoder
            'use_fingerprints': False,
            'use_contrastive_learning': False
        }
        
        key_tasks = ['Carcinogenicity', 'Ames Mutagenicity', 'Acute oral toxicity (LD50)']
        for task in key_tasks:
            if task in CLASSIFICATION_TASKS:
                config['task_configs'][task] = {'output_dim': 1, 'task_type': 'classification'}
            else:
                config['task_configs'][task] = {'output_dim': 1, 'task_type': 'regression'}
        
        print("[FWD_PASS] 3. Creating model...")
        from models.toxd4c import ToxD4C
        model = ToxD4C(config=config, device='cpu')
        model.eval()
        
        print("[FWD_PASS] 4. Preparing data for SMILES 'CCO'...")
        extractor = MolecularFeatureExtractor()
        smiles = "CCO"
        mol = Chem.MolFromSmiles(smiles)
        graph_data = extractor.mol_to_graph(mol)
        
        data = {
            'atom_features': torch.tensor(graph_data['atom_features'], dtype=torch.float32),
            'bond_features': torch.tensor(graph_data['bond_features'], dtype=torch.float32),
            'edge_index': torch.tensor(graph_data['edge_index'], dtype=torch.long),
            'coordinates': torch.tensor(graph_data['coordinates'], dtype=torch.float32),
            'batch': torch.zeros(graph_data['num_atoms'], dtype=torch.long)
        }
        print(f"[FWD_PASS]    Data shapes: atom_features={data['atom_features'].shape}, batch={data['batch'].shape}")

        print("[FWD_PASS] 5. Running forward pass...")
        with torch.no_grad():
            outputs = model(data, [smiles])
        print("[FWD_PASS]    Forward pass complete.")
        
        print("[FWD_PASS] 6. Processing predictions...")
        predictions = outputs['predictions']
        
        print(f"✓ 前向传播成功")
        print(f"  输入分子: {smiles}")
        print(f"  预测任务数: {len(predictions)}")
        
        for task in key_tasks:
            if task in predictions:
                pred = predictions[task]
                print(f"  {task}: {pred.shape} -> {pred.item():.4f}")
        
    except Exception as e:
        import traceback
        print(f"✗ 前向传播失败: {e}")
        traceback.print_exc()


def main():
    """主函数"""
    print("ToxD4C 快速测试")
    print("=" * 50)
    
    # 运行各项测试
    test_config()
    test_data_loading()
    test_feature_extraction()
    test_model_creation()
    test_forward_pass()
    
    print("\n" + "=" * 50)
    print("测试完成!")
    
    print(f"\n总结:")
    print(f"- 支持 {len(CLASSIFICATION_TASKS)} 个分类任务")
    print(f"- 支持 {len(REGRESSION_TASKS)} 个回归任务")
    print(f"- 总共 {len(CLASSIFICATION_TASKS + REGRESSION_TASKS)} 个毒性预测任务")
    print(f"- 框架包含: 配置管理、数据加载、特征提取、模型架构、训练推理")


if __name__ == "__main__":
    main() 