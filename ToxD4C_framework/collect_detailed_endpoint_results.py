#!/usr/bin/env python3
"""
收集所有消融实验在26个分类终点和5个回归终点上的详细结果
"""

import os
import json
import subprocess
import re
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import torch

def _contains_lmdb(dir_path: Path) -> bool:
    return (dir_path / 'train.lmdb').exists() and (dir_path / 'valid.lmdb').exists() and (dir_path / 'test.lmdb').exists()

def resolve_data_dir(user_path: Optional[str]) -> str:
    """解析数据目录，若给定目录不包含所需LMDB，则尝试常见候选路径。"""
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    # 常见数据集位置候选
    candidates.extend([
        Path('ToxD4C') / 'data' / 'data' / 'dataset',
        Path('D4C-new') / 'ToxD4C_en' / 'data' / 'dataset',
        Path('ToxD4C_ECO') / 'data' / 'data' / 'dataset',
    ])

    for p in candidates:
        if p.exists() and _contains_lmdb(p):
            return str(p)

    # 找不到就返回用户原始输入
    return user_path or ''
def find_ablation_experiments() -> List[str]:
    """找到所有消融实验目录。
    同时扫描顶层 experiments 和 ToxD4C/experiments 目录，去重合并。
    """
    candidate_roots = [Path("experiments"), Path("ToxD4C") / "experiments"]
    ablation_dirs: List[str] = []

    for root in candidate_roots:
        if not root.exists():
            continue
        for exp_dir in root.iterdir():
            if exp_dir.is_dir() and "toxd4c_ablation_" in exp_dir.name:
                ablation_dirs.append(str(exp_dir))

    # 去重并排序
    return sorted(list({p for p in ablation_dirs}))

def has_valid_checkpoint(exp_dir: str) -> bool:
    """检查实验是否有有效的检查点"""
    checkpoints_dir = Path(exp_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        return False
    
    # 查找最佳检查点
    best_checkpoints = list(checkpoints_dir.glob("*_best.pth"))
    return len(best_checkpoints) > 0

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

def run_detailed_test_evaluation(exp_dir: str, data_dir: str = "data/data/processed",
                                 split: str = 'test', out_name: Optional[str] = None) -> Optional[Dict]:
    """为指定实验运行详细评估（默认test，可选val），获取每个终点的指标"""
    print(f"正在进行详细评估: {exp_dir}")
    
    # 创建详细评估脚本（位于 ToxD4C 下，便于包导入）
    detailed_eval_script = create_detailed_evaluation_script()
    
    try:
        # 运行详细测试评估
        # 根据可用情况解析数据目录
        data_dir = resolve_data_dir(data_dir)

        cmd = ["python", detailed_eval_script,
               "--experiment_dir", exp_dir,
               "--data_dir", data_dir,
               "--batch_size", "16",
               "--split", split]
        if out_name:
            cmd += ["--out_name", out_name]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"  ❌ 详细评估失败: {result.stderr}")
            return None
        
        # 读取详细测试结果
        # 读取详细测试结果（按 out_name 或 split 推断）
        if out_name:
            detailed_results_path = Path(exp_dir) / "checkpoints" / out_name
        else:
            fname = 'detailed_endpoint_results.json' if split == 'test' else f'detailed_endpoint_results_{split}.json'
            detailed_results_path = Path(exp_dir) / "checkpoints" / fname
        if detailed_results_path.exists():
            with open(detailed_results_path, 'r') as f:
                detailed_results = json.load(f)
            print(f"  ✅ 详细评估成功")
            return detailed_results
        else:
            print(f"  ❌ 未找到详细结果文件")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ 详细评估超时")
        return None
    except Exception as e:
        print(f"  ❌ 详细评估出错: {e}")
        return None

def create_detailed_evaluation_script() -> str:
    """创建详细评估脚本。
    将脚本放在 ToxD4C/ 目录下以复用包内导入路径。
    """
    script_path = str(Path("ToxD4C") / "evaluate_detailed_endpoints.py")

    if Path(script_path).exists():
        return script_path
    
    script_content = '''#!/usr/bin/env python3
"""
详细评估脚本：获取每个终点的具体指标
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score

from data.lmdb_dataset import create_lmdb_dataloaders
from models.toxd4c import ToxD4C

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义任务名称
CLASSIFICATION_TASKS = [
    "Carcinogenicity", "Ames Mutagenicity", "Respiratory toxicity", "Eye irritation", 
    "Eye corrosion", "Cardiotoxicity1", "Cardiotoxicity10", "Cardiotoxicity30", 
    "Cardiotoxicity5", "CYP1A2", "CYP2C19", "CYP2C9", "CYP2D6", "CYP3A4", 
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

REGRESSION_TASKS = [
    "Acute oral toxicity (LD50)", "LC50DM", "BCF", "LC50", "IGC50"
]

def _safe_loss(pred, target, mask, loss_fn, device):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    return loss_fn(pred[mask], target[mask])

@torch.no_grad()
def evaluate_detailed_endpoints(model: torch.nn.Module,
                               dataloader: torch.utils.data.DataLoader,
                               device: torch.device,
                               config: Dict) -> Dict:
    model.eval()
    
    all_cls_preds = []
    all_cls_targets = []
    all_cls_masks = []
    all_reg_preds = []
    all_reg_targets = []
    all_reg_masks = []

    for batch in dataloader:
        if batch is None:
            continue

        data = {
            'atom_features': batch['atom_features'].to(device),
            'edge_index': batch['edge_index'].to(device),
            'coordinates': batch['coordinates'].to(device),
            'batch': batch['batch'].to(device)
        }
        cls_labels = batch['classification_labels'].to(device)
        reg_labels = batch['regression_labels'].to(device)
        cls_mask = batch['classification_mask'].to(device)
        reg_mask = batch['regression_mask'].to(device)

        outputs = model(data, batch['smiles'])
        cls_logits = outputs['predictions']['classification']
        reg_preds = outputs['predictions']['regression']

        # 收集预测和标签
        has_cls = cls_logits.numel() > 0
        has_reg = reg_preds.numel() > 0

        if has_cls:
            cls_probs = torch.sigmoid(cls_logits)
            all_cls_preds.append(cls_probs.cpu())
            all_cls_targets.append(cls_labels.cpu())
            all_cls_masks.append(cls_mask.cpu())

        if has_reg:
            all_reg_preds.append(reg_preds.cpu())
            all_reg_targets.append(reg_labels.cpu())
            all_reg_masks.append(reg_mask.cpu())

    # 计算每个终点的详细指标
    results = {
        'classification_endpoints': {},
        'regression_endpoints': {},
        'summary': {}
    }
    
    # 分类任务详细指标
    if all_cls_preds:
        cls_preds = torch.cat(all_cls_preds, dim=0).numpy()
        cls_tgts = torch.cat(all_cls_targets, dim=0).numpy()
        cls_masks = torch.cat(all_cls_masks, dim=0).numpy()
        
        cls_metrics = []
        
        for i, task_name in enumerate(CLASSIFICATION_TASKS[:cls_preds.shape[1]]):
            mask = cls_masks[:, i].astype(bool)
            if mask.sum() == 0:
                continue
                
            y_score = cls_preds[mask, i]
            y_true = cls_tgts[mask, i].astype(int)
            y_pred = (y_score > 0.5).astype(int)
            
            task_metrics = {
                'task_index': i,
                'task_name': task_name,
                'n_samples': int(mask.sum()),
                'n_positive': int(y_true.sum()),
                'n_negative': int((y_true == 0).sum())
            }
            
            try:
                task_metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                task_metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
                task_metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
                task_metrics['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
                
                if len(np.unique(y_true)) > 1:
                    task_metrics['auc'] = float(roc_auc_score(y_true, y_score))
                else:
                    task_metrics['auc'] = None
                    
            except Exception as e:
                logger.warning(f"Error computing metrics for {task_name}: {e}")
                task_metrics.update({
                    'accuracy': None, 'precision': None, 'recall': None, 
                    'f1_score': None, 'auc': None
                })
            
            results['classification_endpoints'][task_name] = task_metrics
            cls_metrics.append(task_metrics)
        
        # 分类汇总
        valid_accs = [m['accuracy'] for m in cls_metrics if m['accuracy'] is not None]
        valid_aucs = [m['auc'] for m in cls_metrics if m['auc'] is not None]
        valid_f1s = [m['f1_score'] for m in cls_metrics if m['f1_score'] is not None]
        
        results['summary']['classification'] = {
            'avg_accuracy': float(np.mean(valid_accs)) if valid_accs else None,
            'avg_auc': float(np.mean(valid_aucs)) if valid_aucs else None,
            'avg_f1_score': float(np.mean(valid_f1s)) if valid_f1s else None,
            'n_tasks_evaluated': len(cls_metrics)
        }
    
    # 回归任务详细指标
    if all_reg_preds:
        reg_preds = torch.cat(all_reg_preds, dim=0).numpy()
        reg_tgts = torch.cat(all_reg_targets, dim=0).numpy()
        reg_masks = torch.cat(all_reg_masks, dim=0).numpy()
        
        reg_metrics = []
        
        for i, task_name in enumerate(REGRESSION_TASKS[:reg_preds.shape[1]]):
            mask = reg_masks[:, i].astype(bool)
            if mask.sum() == 0:
                continue
                
            y_pred = reg_preds[mask, i]
            y_true = reg_tgts[mask, i]
            
            task_metrics = {
                'task_index': i,
                'task_name': task_name,
                'n_samples': int(mask.sum()),
                'y_true_mean': float(np.mean(y_true)),
                'y_true_std': float(np.std(y_true)),
                'y_pred_mean': float(np.mean(y_pred)),
                'y_pred_std': float(np.std(y_pred))
            }
            
            try:
                mse = mean_squared_error(y_true, y_pred)
                task_metrics['mse'] = float(mse)
                task_metrics['rmse'] = float(np.sqrt(mse))
                task_metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
                
                if np.var(y_true) < 1e-6:
                    task_metrics['r2'] = 0.0
                else:
                    task_metrics['r2'] = float(r2_score(y_true, y_pred))
                    
            except Exception as e:
                logger.warning(f"Error computing metrics for {task_name}: {e}")
                task_metrics.update({
                    'mse': None, 'rmse': None, 'mae': None, 'r2': None
                })
            
            results['regression_endpoints'][task_name] = task_metrics
            reg_metrics.append(task_metrics)
        
        # 回归汇总
        valid_r2s = [m['r2'] for m in reg_metrics if m['r2'] is not None]
        valid_rmses = [m['rmse'] for m in reg_metrics if m['rmse'] is not None]
        valid_maes = [m['mae'] for m in reg_metrics if m['mae'] is not None]
        
        results['summary']['regression'] = {
            'avg_r2': float(np.mean(valid_r2s)) if valid_r2s else None,
            'avg_rmse': float(np.mean(valid_rmses)) if valid_rmses else None,
            'avg_mae': float(np.mean(valid_maes)) if valid_maes else None,
            'n_tasks_evaluated': len(reg_metrics)
        }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='详细评估每个终点的性能')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_atoms', type=int, default=64)
    parser.add_argument('--split', type=str, choices=['test','val'], default='test')
    parser.add_argument('--out_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # 解析检查点和配置
    exp_dir = Path(args.experiment_dir)
    ckpt_dir = exp_dir / 'checkpoints'
    
    # 查找最佳检查点
    best_checkpoints = list(ckpt_dir.glob('*_best.pth'))
    if not best_checkpoints:
        raise FileNotFoundError(f'No best checkpoint found in {ckpt_dir}')
    ckpt_path = best_checkpoints[0]
    
    config_path = ckpt_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 构建模型并加载权重
    model = ToxD4C(config=config, device=device).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载数据
    train_loader, val_loader, test_loader = create_lmdb_dataloaders(args.data_dir,
                                               batch_size=args.batch_size,
                                               max_atoms=args.max_atoms)

    # 选择评估划分
    if args.split == 'val':
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    # 详细评估
    results = evaluate_detailed_endpoints(model, eval_loader, device, config)

    # 保存结果
    if args.out_name:
        output_path = ckpt_dir / args.out_name
    else:
        output_path = ckpt_dir / ('detailed_endpoint_results.json' if args.split == 'test' else f'detailed_endpoint_results_{args.split}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f'详细终点结果已保存到: {output_path}')
    
    # 打印摘要
    if 'classification' in results['summary']:
        cls_summary = results['summary']['classification']
        print(f"分类任务摘要 ({cls_summary.get('n_tasks_evaluated', 0)} 个任务):")
        if cls_summary.get('avg_accuracy') is not None:
            print(f"  平均准确率: {cls_summary['avg_accuracy']:.4f}")
        if cls_summary.get('avg_auc') is not None:
            print(f"  平均AUC: {cls_summary['avg_auc']:.4f}")
        if cls_summary.get('avg_f1_score') is not None:
            print(f"  平均F1: {cls_summary['avg_f1_score']:.4f}")
    
    if 'regression' in results['summary']:
        reg_summary = results['summary']['regression']
        print(f"回归任务摘要 ({reg_summary.get('n_tasks_evaluated', 0)} 个任务):")
        if reg_summary.get('avg_r2') is not None:
            print(f"  平均R²: {reg_summary['avg_r2']:.4f}")
        if reg_summary.get('avg_rmse') is not None:
            print(f"  平均RMSE: {reg_summary['avg_rmse']:.4f}")
        if reg_summary.get('avg_mae') is not None:
            print(f"  平均MAE: {reg_summary['avg_mae']:.4f}")

if __name__ == '__main__':
    main()
'''
    
    # 确保目录存在
    Path(script_path).parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def create_endpoint_comparison_report(all_results: List[Dict]) -> pd.DataFrame:
    """创建终点对比报告"""
    
    # 获取所有任务名称
    all_cls_tasks = set()
    all_reg_tasks = set()
    
    for result in all_results:
        if 'detailed_results' in result and result['detailed_results']:
            cls_endpoints = result['detailed_results'].get('classification_endpoints', {})
            reg_endpoints = result['detailed_results'].get('regression_endpoints', {})
            all_cls_tasks.update(cls_endpoints.keys())
            all_reg_tasks.update(reg_endpoints.keys())
    
    all_cls_tasks = sorted(list(all_cls_tasks))
    all_reg_tasks = sorted(list(all_reg_tasks))
    
    # 创建分类任务对比表
    cls_data = []
    for task in all_cls_tasks:
        row = {'Task': task, 'Type': 'Classification'}
        for result in all_results:
            exp_name = result['experiment_name']
            if ('detailed_results' in result and result['detailed_results'] and 
                task in result['detailed_results'].get('classification_endpoints', {})):
                metrics = result['detailed_results']['classification_endpoints'][task]
                row[f'{exp_name}_Accuracy'] = metrics.get('accuracy')
                row[f'{exp_name}_AUC'] = metrics.get('auc')
                row[f'{exp_name}_F1'] = metrics.get('f1_score')
                row[f'{exp_name}_Samples'] = metrics.get('n_samples')
            else:
                row[f'{exp_name}_Accuracy'] = None
                row[f'{exp_name}_AUC'] = None
                row[f'{exp_name}_F1'] = None
                row[f'{exp_name}_Samples'] = None
        cls_data.append(row)
    
    # 创建回归任务对比表
    reg_data = []
    for task in all_reg_tasks:
        row = {'Task': task, 'Type': 'Regression'}
        for result in all_results:
            exp_name = result['experiment_name']
            if ('detailed_results' in result and result['detailed_results'] and 
                task in result['detailed_results'].get('regression_endpoints', {})):
                metrics = result['detailed_results']['regression_endpoints'][task]
                row[f'{exp_name}_R2'] = metrics.get('r2')
                row[f'{exp_name}_RMSE'] = metrics.get('rmse')
                row[f'{exp_name}_MAE'] = metrics.get('mae')
                row[f'{exp_name}_Samples'] = metrics.get('n_samples')
            else:
                row[f'{exp_name}_R2'] = None
                row[f'{exp_name}_RMSE'] = None
                row[f'{exp_name}_MAE'] = None
                row[f'{exp_name}_Samples'] = None
        reg_data.append(row)
    
    # 合并数据
    all_data = cls_data + reg_data
    return pd.DataFrame(all_data)

def main():
    print("🔍 查找所有消融实验...")
    ablation_experiments = find_ablation_experiments()
    print(f"找到 {len(ablation_experiments)} 个消融实验")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['test','val'], default='test', help='选择评估划分')
    parser.add_argument('--agg_suffix', type=str, default='', help='输出文件名后缀（如 _val）')
    parser.add_argument('--data_dir', type=str, default='data/data/processed')
    parser.add_argument('--out_dir', type=str, default='.', help='聚合结果输出目录（默认当前目录）')
    parser.add_argument('--only', type=str, default=None, help='仅评估名称包含这些关键词的实验（逗号分隔）')
    args = parser.parse_args()

    # 可选过滤实验
    if args.only:
        filters = [s.strip() for s in args.only.split(',') if s.strip()]
        ablation_experiments = [p for p in ablation_experiments if any(f in p for f in filters)]
        print(f"按过滤条件筛选后，剩余 {len(ablation_experiments)} 个实验")

    all_results = []
    
    for exp_dir in ablation_experiments:
        print(f"\n📊 处理实验: {exp_dir}")
        
        # 检查是否有有效检查点
        if not has_valid_checkpoint(exp_dir):
            print(f"  ⚠️  跳过 - 没有有效检查点")
            continue
        
        # 检查是否已有详细结果
        # 按选择 split 的默认输出名寻找现有文件
        def_name = 'detailed_endpoint_results.json' if args.split=='test' else f'detailed_endpoint_results_{args.split}.json'
        detailed_results_path = Path(exp_dir) / "checkpoints" / def_name
        
        if detailed_results_path.exists():
            print(f"  📁 使用现有详细结果")
            with open(detailed_results_path, 'r') as f:
                detailed_results = json.load(f)
        else:
            # 运行详细评估
            detailed_results = run_detailed_test_evaluation(exp_dir, data_dir=args.data_dir, split=args.split)
            if detailed_results is None:
                print(f"  ⚠️  跳过 - 详细评估失败")
                continue
        
        # 加载配置
        config = load_config(exp_dir)
        
        # 整理结果
        result = {
            'experiment_name': extract_experiment_name(exp_dir),
            'experiment_dir': exp_dir,
            'config': config,
            'detailed_results': detailed_results,
        }
        
        all_results.append(result)
        
        # 显示摘要
        if detailed_results and 'summary' in detailed_results:
            summary = detailed_results['summary']
            if 'classification' in summary:
                cls_summary = summary['classification']
                print(f"  📊 分类: {cls_summary['n_tasks_evaluated']}个任务, "
                      f"平均准确率: {cls_summary['avg_accuracy']:.4f}")
            if 'regression' in summary:
                reg_summary = summary['regression']
                print(f"  📈 回归: {reg_summary['n_tasks_evaluated']}个任务, "
                      f"平均R²: {reg_summary['avg_r2']:.4f}")
        
        print(f"  ✅ 详细结果收集完成")
    
    print(f"\n📈 生成终点对比报告...")
    
    # 创建详细结果JSON
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f'detailed_endpoint_results_all_experiments{args.agg_suffix}.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"详细终点结果已保存到: {out_json}")
    
    # 创建终点对比表格
    df = create_endpoint_comparison_report(all_results)
    out_csv = out_dir / f'endpoint_comparison_all_experiments{args.agg_suffix}.csv'
    df.to_csv(out_csv, index=False)
    print(f"终点对比表格已保存到: {out_csv}")
    
    print(f"\n✅ 所有终点详细结果收集完成!")
    print(f"📊 详细结果文件: {out_json}")
    print(f"📋 对比表格文件: {out_csv}")

if __name__ == "__main__":
    main()
