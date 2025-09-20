#!/usr/bin/env python3
"""
收集所有消融实验的详细结果，包括模型参数量、训练时间等信息
"""

import os
import json
import subprocess
import re
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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

def extract_model_params_from_log(exp_dir: str) -> Optional[int]:
    """从训练日志中提取模型参数量"""
    log_path = Path(exp_dir) / "train.log"
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # 查找模型参数量信息
        patterns = [
            r"Total parameters:\s*([0-9,]+)",
            r"Model created successfully\. Total parameters:\s*([0-9,]+)",
            r"total_params.*?([0-9,]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                param_str = match.group(1).replace(',', '')
                return int(param_str)
        
        return None
    except Exception as e:
        print(f"  ⚠️  无法从日志提取参数量: {e}")
        return None

def extract_training_info_from_log(exp_dir: str) -> Dict:
    """从训练日志中提取训练信息"""
    log_path = Path(exp_dir) / "train.log"
    info = {
        'total_epochs': None,
        'best_epoch': None,
        'training_time': None,
        'final_val_acc': None,
        'final_val_auc': None,
        'final_val_r2': None
    }
    
    if not log_path.exists():
        return info
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # 提取最终验证结果
        val_acc_pattern = r"Average Classification Accuracy:\s*([0-9.]+)"
        val_auc_pattern = r"Average AUC:\s*([0-9.]+)"
        val_r2_pattern = r"Average R²:\s*([0-9.-]+)"
        
        val_acc_matches = re.findall(val_acc_pattern, content)
        val_auc_matches = re.findall(val_auc_pattern, content)
        val_r2_matches = re.findall(val_r2_pattern, content)
        
        if val_acc_matches:
            info['final_val_acc'] = float(val_acc_matches[-1])  # 取最后一个（最终结果）
        if val_auc_matches:
            info['final_val_auc'] = float(val_auc_matches[-1])
        if val_r2_matches:
            info['final_val_r2'] = float(val_r2_matches[-1])
        
        # 提取训练轮数信息
        epoch_pattern = r"Epoch (\d+)/(\d+)"
        epoch_matches = re.findall(epoch_pattern, content)
        if epoch_matches:
            last_epoch, total_epochs = epoch_matches[-1]
            info['total_epochs'] = int(total_epochs)
            info['best_epoch'] = int(last_epoch)
        
        # 提取早停信息
        early_stop_pattern = r"Early stopping"
        if re.search(early_stop_pattern, content):
            info['early_stopped'] = True
        else:
            info['early_stopped'] = False
            
        return info
    except Exception as e:
        print(f"  ⚠️  无法从日志提取训练信息: {e}")
        return info

def extract_experiment_timestamp(exp_dir: str) -> Optional[str]:
    """从实验目录名提取时间戳"""
    dir_name = Path(exp_dir).name
    # 匹配时间戳格式 YYYYMMDD_HHMMSS
    timestamp_pattern = r"(\d{8}_\d{6})"
    match = re.search(timestamp_pattern, dir_name)
    if match:
        return match.group(1)
    return None

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
    dir_name = Path(exp_dir).name
    
    # 移除时间戳
    parts = dir_name.split('_')
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        name_parts = parts[:-2]
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

def load_training_results(exp_dir: str) -> Dict:
    """加载训练结果"""
    # 查找训练结果文件
    checkpoints_dir = Path(exp_dir) / "checkpoints"
    result_files = list(checkpoints_dir.glob("*_results.json"))
    
    if result_files:
        with open(result_files[0], 'r') as f:
            return json.load(f)
    return {}

def create_comprehensive_report(results: List[Dict]) -> pd.DataFrame:
    """创建综合结果报告"""
    report_data = []
    
    for result in results:
        exp_name = result['experiment_name']
        config = result.get('config', {})
        test_metrics = result.get('test_metrics', {})
        training_info = result.get('training_info', {})
        training_results = result.get('training_results', {})
        
        # 基本信息
        row = {
            'Experiment': exp_name,
            'Timestamp': result.get('timestamp'),
            'Model_Parameters': result.get('model_params'),
        }
        
        # 测试集指标
        row['Test_Total_Loss'] = result.get('total_loss')
        row['Test_Classification_Loss'] = result.get('classification_loss')
        row['Test_Regression_Loss'] = result.get('regression_loss')
        
        # 测试集分类指标
        if 'avg_cls_accuracy' in test_metrics:
            row['Test_Avg_Classification_Accuracy'] = test_metrics['avg_cls_accuracy']
        if 'avg_auc' in test_metrics:
            row['Test_Avg_AUC'] = test_metrics['avg_auc']
        
        # 测试集回归指标
        if 'avg_r2' in test_metrics:
            row['Test_Avg_R2'] = test_metrics['avg_r2']
        if 'avg_rmse' in test_metrics:
            row['Test_Avg_RMSE'] = test_metrics['avg_rmse']
        
        # 验证集指标（来自训练日志）
        row['Val_Avg_Classification_Accuracy'] = training_info.get('final_val_acc')
        row['Val_Avg_AUC'] = training_info.get('final_val_auc')
        row['Val_Avg_R2'] = training_info.get('final_val_r2')
        
        # 训练信息
        row['Total_Epochs'] = training_info.get('total_epochs')
        row['Best_Epoch'] = training_info.get('best_epoch')
        row['Early_Stopped'] = training_info.get('early_stopped')
        
        # 配置信息
        row['Use_GNN'] = config.get('use_gnn')
        row['Use_Transformer'] = config.get('use_transformer')
        row['Use_Geometric'] = config.get('use_geometric_encoder')
        row['Use_Hierarchical'] = config.get('use_hierarchical_encoder')
        row['Use_Fingerprints'] = config.get('use_fingerprints')
        row['Use_Contrastive'] = config.get('use_contrastive_learning')
        row['Enable_Classification'] = config.get('enable_classification')
        row['Enable_Regression'] = config.get('enable_regression')
        row['Fusion_Method'] = config.get('fusion_method')
        row['GNN_Backbone'] = config.get('gnn_backbone')
        row['Hidden_Dim'] = config.get('hidden_dim')
        row['Num_Encoder_Layers'] = config.get('num_encoder_layers')
        row['Num_Attention_Heads'] = config.get('num_attention_heads')
        row['Dropout'] = config.get('dropout')
        row['Learning_Rate'] = config.get('learning_rate')
        row['Batch_Size'] = config.get('batch_size')
        
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
        
        # 加载训练结果
        training_results = load_training_results(exp_dir)
        
        # 提取模型参数量
        model_params = extract_model_params_from_log(exp_dir)
        if model_params is None and 'model_params' in training_results:
            model_params = training_results['model_params']
        
        # 提取训练信息
        training_info = extract_training_info_from_log(exp_dir)
        
        # 提取时间戳
        timestamp = extract_experiment_timestamp(exp_dir)
        
        # 整理结果
        result = {
            'experiment_name': extract_experiment_name(exp_dir),
            'experiment_dir': exp_dir,
            'timestamp': timestamp,
            'config': config,
            'test_metrics': test_results.get('metrics', {}),
            'total_loss': test_results.get('total_loss'),
            'classification_loss': test_results.get('classification_loss'),
            'regression_loss': test_results.get('regression_loss'),
            'training_results': training_results,
            'training_info': training_info,
            'model_params': model_params,
        }
        
        all_results.append(result)
        print(f"  ✅ 结果收集完成 (参数量: {model_params:,} 个)" if model_params else "  ✅ 结果收集完成")
    
    print(f"\n📈 生成综合报告...")
    
    # 创建详细结果JSON
    with open('enhanced_ablation_results_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"详细结果已保存到: enhanced_ablation_results_detailed.json")
    
    # 创建表格报告
    df = create_comprehensive_report(all_results)
    
    # 保存为CSV
    df.to_csv('enhanced_ablation_results_summary.csv', index=False)
    print(f"表格报告已保存到: enhanced_ablation_results_summary.csv")
    
    # 显示摘要
    print(f"\n📋 实验摘要:")
    print(f"总实验数: {len(all_results)}")
    
    # 按性能排序显示前几名
    if len(all_results) > 0:
        print(f"\n🏆 按测试集分类准确率排序的前5名:")
        df_sorted = df.sort_values('Test_Avg_Classification_Accuracy', ascending=False)
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows()):
            acc = row['Test_Avg_Classification_Accuracy']
            auc = row['Test_Avg_AUC']
            r2 = row['Test_Avg_R2']
            params = row['Model_Parameters']
            print(f"{i+1}. {row['Experiment']}")
            print(f"   测试准确率: {acc:.4f}, AUC: {auc:.4f}, R²: {r2:.4f}")
            print(f"   模型参数: {params:,} 个" if params else "   模型参数: 未知")
        
        print(f"\n🔬 按测试集R²排序的前5名:")
        df_r2_sorted = df.sort_values('Test_Avg_R2', ascending=False)
        for i, (_, row) in enumerate(df_r2_sorted.head(5).iterrows()):
            r2 = row['Test_Avg_R2']
            rmse = row['Test_Avg_RMSE']
            acc = row['Test_Avg_Classification_Accuracy']
            params = row['Model_Parameters']
            print(f"{i+1}. {row['Experiment']}")
            print(f"   R²: {r2:.4f}, RMSE: {rmse:.4f}, 准确率: {acc:.4f}")
            print(f"   模型参数: {params:,} 个" if params else "   模型参数: 未知")
    
    print(f"\n✅ 所有结果收集完成!")
    print(f"📊 详细结果文件: enhanced_ablation_results_detailed.json")
    print(f"📋 表格摘要文件: enhanced_ablation_results_summary.csv")

if __name__ == "__main__":
    main()
