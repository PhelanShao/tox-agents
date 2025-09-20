#!/usr/bin/env python3
"""
收集所有消融实验的测试集结果并生成综合报告
"""

import os
import json
import subprocess
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

def find_ablation_experiments() -> List[str]:
    """找到所有消融实验目录"""
    experiments_dir = Path("experiments")
    ablation_dirs = []
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and "toxd4c_ablation_" in exp_dir.name:
            ablation_dirs.append(str(exp_dir))
    
    return sorted(ablation_dirs)

def has_valid_checkpoint(exp_dir: str) -> bool:
    """检查实验是否有有效的检查点"""
    checkpoints_dir = Path(exp_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        return False
    
    # 查找最佳检查点
    best_checkpoints = list(checkpoints_dir.glob("*_best.pth"))
    return len(best_checkpoints) > 0

def run_test_evaluation(exp_dir: str, data_dir: str = "data/data/processed") -> Optional[Dict]:
    """为指定实验运行测试评估"""
    print(f"正在评估实验: {exp_dir}")
    
    try:
        # 运行测试评估
        cmd = [
            "python", "evaluate_test.py",
            "--experiment_dir", exp_dir,
            "--data_dir", data_dir,
            "--batch_size", "16"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"  ❌ 评估失败: {result.stderr}")
            return None
        
        # 读取测试结果
        test_results_path = Path(exp_dir) / "checkpoints" / "test_results.json"
        if test_results_path.exists():
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)
            print(f"  ✅ 评估成功")
            return test_results
        else:
            print(f"  ❌ 未找到测试结果文件")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ 评估超时")
        return None
    except Exception as e:
        print(f"  ❌ 评估出错: {e}")
        return None

def extract_experiment_name(exp_dir: str) -> str:
    """从实验目录提取实验名称"""
    # 从路径中提取实验名称
    dir_name = Path(exp_dir).name
    
    # 移除时间戳
    parts = dir_name.split('_')
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        name_parts = parts[:-2]  # 移除最后两个时间戳部分
    else:
        name_parts = parts
    
    return '_'.join(name_parts)

def load_config(exp_dir: str) -> Dict:
    """加载实验配置"""
    config_path = Path(exp_dir) / "checkpoints" / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def create_comprehensive_report(results: List[Dict]) -> pd.DataFrame:
    """创建综合结果报告"""
    report_data = []
    
    for result in results:
        exp_name = result['experiment_name']
        config = result.get('config', {})
        metrics = result.get('test_metrics', {})
        
        # 基本信息
        row = {
            'Experiment': exp_name,
            'Total_Loss': result.get('total_loss', None),
            'Classification_Loss': result.get('classification_loss', None),
            'Regression_Loss': result.get('regression_loss', None),
        }
        
        # 分类指标
        if 'avg_cls_accuracy' in metrics:
            row['Avg_Classification_Accuracy'] = metrics['avg_cls_accuracy']
        if 'avg_auc' in metrics:
            row['Avg_AUC'] = metrics['avg_auc']
        
        # 回归指标
        if 'avg_r2' in metrics:
            row['Avg_R2'] = metrics['avg_r2']
        if 'avg_rmse' in metrics:
            row['Avg_RMSE'] = metrics['avg_rmse']
        
        # 配置信息
        row['Use_GNN'] = config.get('use_gnn', None)
        row['Use_Transformer'] = config.get('use_transformer', None)
        row['Use_Geometric'] = config.get('use_geometric_encoder', None)
        row['Use_Hierarchical'] = config.get('use_hierarchical_encoder', None)
        row['Use_Fingerprints'] = config.get('use_fingerprints', None)
        row['Use_Contrastive'] = config.get('use_contrastive_learning', None)
        row['Enable_Classification'] = config.get('enable_classification', None)
        row['Enable_Regression'] = config.get('enable_regression', None)
        row['Fusion_Method'] = config.get('fusion_method', None)
        row['GNN_Backbone'] = config.get('gnn_backbone', None)
        
        report_data.append(row)
    
    return pd.DataFrame(report_data)

def main():
    print("🔍 查找所有消融实验...")
    ablation_experiments = find_ablation_experiments()
    print(f"找到 {len(ablation_experiments)} 个消融实验")
    
    all_results = []
    
    for exp_dir in ablation_experiments:
        print(f"\n📊 处理实验: {exp_dir}")
        
        # 检查是否有有效检查点
        if not has_valid_checkpoint(exp_dir):
            print(f"  ⚠️  跳过 - 没有有效检查点")
            continue
        
        # 检查是否已有测试结果
        test_results_path = Path(exp_dir) / "checkpoints" / "test_results.json"
        
        if test_results_path.exists():
            print(f"  📁 使用现有测试结果")
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)
        else:
            # 运行测试评估
            test_results = run_test_evaluation(exp_dir)
            if test_results is None:
                continue
        
        # 加载配置
        config = load_config(exp_dir)
        
        # 整理结果
        result = {
            'experiment_name': extract_experiment_name(exp_dir),
            'experiment_dir': exp_dir,
            'config': config,
            'test_metrics': test_results.get('metrics', {}),
            'total_loss': test_results.get('total_loss'),
            'classification_loss': test_results.get('classification_loss'),
            'regression_loss': test_results.get('regression_loss'),
        }
        
        all_results.append(result)
        print(f"  ✅ 结果收集完成")
    
    print(f"\n📈 生成综合报告...")
    
    # 创建详细结果JSON
    with open('ablation_results_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"详细结果已保存到: ablation_results_detailed.json")
    
    # 创建表格报告
    df = create_comprehensive_report(all_results)
    
    # 保存为CSV
    df.to_csv('ablation_results_summary.csv', index=False)
    print(f"表格报告已保存到: ablation_results_summary.csv")
    
    # 显示摘要
    print(f"\n📋 实验摘要:")
    print(f"总实验数: {len(all_results)}")
    
    # 按性能排序显示前几名
    if len(all_results) > 0:
        print(f"\n🏆 按分类准确率排序的前5名:")
        df_sorted = df.sort_values('Avg_Classification_Accuracy', ascending=False)
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
            acc = row['Avg_Classification_Accuracy']
            auc = row['Avg_AUC']
            r2 = row['Avg_R2']
            print(f"{i+1}. {row['Experiment']}")
            print(f"   分类准确率: {acc:.4f}, AUC: {auc:.4f}, R²: {r2:.4f}")
    
    print(f"\n✅ 所有结果收集完成!")

if __name__ == "__main__":
    main()
