#!/usr/bin/env python3
"""
详细评估脚本：获取每个终点的具体指标
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from data.lmdb_dataset import create_lmdb_dataloaders
from models.toxd4c import ToxD4C

# Ensure local imports work when run as a script from repo root
import sys as _sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.append(str(_HERE))
from metrics.uncertainty_ad import (
    compute_ece,
    compute_brier_score,
    reliability_curve,
    plot_reliability_diagram,
    tta_predict,
    collect_embeddings,
    build_ad_from_embeddings,
    summarize_prediction_variance,
)

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
                               config: Dict,
                               *,
                               with_uncertainty: bool = False,
                               tta_runs: int = 1,
                               tta_noise_std: float = 0.02,
                               mc_dropout: bool = False) -> Dict:
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

        # TTA path if requested
        if with_uncertainty and (tta_runs > 1 or mc_dropout):
            cls_all, reg_all = tta_predict(
                model,
                {
                    'atom_features': batch['atom_features'],
                    'edge_index': batch['edge_index'],
                    'coordinates': batch['coordinates'],
                    'batch': batch['batch'],
                },
                batch['smiles'],
                runs=tta_runs,
                coord_noise_std=tta_noise_std,
                device=device,
                mc_dropout=mc_dropout,
            )
            if cls_all is not None:
                cls_probs = torch.tensor(cls_all.mean(axis=0))
                all_cls_preds.append(cls_probs.cpu())
                all_cls_targets.append(cls_labels.cpu())
                all_cls_masks.append(cls_mask.cpu())
                # keep per-run arrays for variance later
                if 'tta_cls_full' not in locals():
                    tta_cls_full: List[np.ndarray] = []
                tta_cls_full.append(cls_all)
            if reg_all is not None:
                reg_mean = torch.tensor(reg_all.mean(axis=0))
                all_reg_preds.append(reg_mean.cpu())
                all_reg_targets.append(reg_labels.cpu())
                all_reg_masks.append(reg_mask.cpu())
                if 'tta_reg_full' not in locals():
                    tta_reg_full: List[np.ndarray] = []
                tta_reg_full.append(reg_all)
            continue

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

                if with_uncertainty:
                    task_metrics['brier'] = compute_brier_score(y_true, y_score)
                    task_metrics['ece'] = compute_ece(y_true, y_score, n_bins=15)
                    
            except Exception as e:
                logger.warning(f"Error computing metrics for {task_name}: {e}")
                task_metrics.update({
                    'accuracy': None, 'precision': None, 'recall': None, 
                    'f1_score': None, 'auc': None,
                    **({'brier': None, 'ece': None} if with_uncertainty else {})
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
        if with_uncertainty:
            valid_brier = [m['brier'] for m in cls_metrics if m.get('brier') is not None]
            valid_ece = [m['ece'] for m in cls_metrics if m.get('ece') is not None]
            results['summary']['classification'].update({
                'avg_brier': float(np.mean(valid_brier)) if valid_brier else None,
                'avg_ece': float(np.mean(valid_ece)) if valid_ece else None,
            })
    
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
    # Uncertainty attachments
    if with_uncertainty:
        unc: Dict[str, Dict] = {}
        if 'tta_cls_full' in locals():
            try:
                cls_full = np.concatenate(tta_cls_full, axis=1)  # [runs, N_total, C]
                var_per_sample = summarize_prediction_variance(cls_full)  # [N_total]
                unc['classification'] = {
                    'per_sample_avg_variance': var_per_sample.tolist(),
                }
            except Exception as e:
                logger.warning(f"Failed to summarize classification variance: {e}")
        if 'tta_reg_full' in locals():
            try:
                reg_full = np.concatenate(tta_reg_full, axis=1)  # [runs, N_total, R]
                var_per_sample = summarize_prediction_variance(reg_full)
                unc['regression'] = {
                    'per_sample_avg_variance': var_per_sample.tolist(),
                }
            except Exception as e:
                logger.warning(f"Failed to summarize regression variance: {e}")
        results['uncertainty'] = unc

        # Calibration curves (for ALL classification tasks)
        try:
            if all_cls_preds:
                cls_preds_np = torch.cat(all_cls_preds, dim=0).numpy()
                cls_tgts_np = torch.cat(all_cls_targets, dim=0).numpy()
                cls_masks_np = torch.cat(all_cls_masks, dim=0).numpy()
                cal_curves = {}
                for i in range(cls_preds_np.shape[1]):
                    mask = cls_masks_np[:, i].astype(bool)
                    y_score = cls_preds_np[mask, i]
                    y_true = cls_tgts_np[mask, i].astype(int)
                    bc, fp, ct = reliability_curve(y_true, y_score, n_bins=15)
                    cal_curves[CLASSIFICATION_TASKS[i]] = {
                        'bin_centers': bc.tolist(),
                        'frac_pos': np.nan_to_num(fp, nan=0.0).tolist(),
                        'counts': [int(x) for x in ct.tolist()],
                    }
                results['calibration_curves'] = cal_curves
        except Exception as e:
            logger.warning(f"Failed to compute calibration curves: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description='详细评估每个终点的性能')
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_atoms', type=int, default=64)
    parser.add_argument('--split', type=str, choices=['test', 'val'], default='test',
                        help='选择评估的数据划分，默认 test')
    parser.add_argument('--out_name', type=str, default=None,
                        help='输出文件名（相对 checkpoints/）。未提供则根据 split 自动命名')
    # 不确定性与校准
    parser.add_argument('--with_uncertainty', action='store_true', help='计算不确定性与校准（ECE、Brier、曲线）')
    parser.add_argument('--tta_runs', type=int, default=1, help='TTA运行次数（>1 启用）')
    parser.add_argument('--tta_noise_std', type=float, default=0.02, help='TTA坐标高斯扰动标准差')
    parser.add_argument('--mc_dropout', action='store_true', help='TTA时启用MC-Dropout（将模型设置为train模式）')
    # 适用域
    parser.add_argument('--with_ad', action='store_true', help='计算适用域（嵌入空间马氏距离）')
    parser.add_argument('--ad_threshold_percentile', type=float, default=95.0, help='马氏距离阈值分位点')
    parser.add_argument('--save_plots', action='store_true', help='保存校准曲线和AD可视化图像')
    
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
    train_loader, val_loader, test_loader = create_lmdb_dataloaders(
        args.data_dir, batch_size=args.batch_size, max_atoms=args.max_atoms)

    # 选择评估 split
    loader = val_loader if args.split == 'val' else test_loader

    # 详细评估
    results = evaluate_detailed_endpoints(
        model, loader, device, config,
        with_uncertainty=args.with_uncertainty,
        tta_runs=args.tta_runs,
        tta_noise_std=args.tta_noise_std,
        mc_dropout=args.mc_dropout,
    )

    # 适用域（基于嵌入的马氏距离）
    ad_report = None
    if args.with_ad:
        try:
            train_embs, _ = collect_embeddings(model, train_loader, device)
            eval_embs, _ = collect_embeddings(model, loader, device)
            if train_embs.size and eval_embs.size:
                ad = build_ad_from_embeddings(train_embs, eval_embs, threshold_percentile=args.ad_threshold_percentile)
                ad_report = {
                    'method': ad.method,
                    'threshold': ad.threshold,
                    'scores': ad.scores,
                    'ood_flags': ad.ood_flags,
                }
                results['applicability_domain'] = ad_report
            else:
                logger.warning('Empty embeddings collected; skipping AD computation')
        except Exception as e:
            logger.warning(f'Failed AD computation: {e}')

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
        if args.with_uncertainty:
            if cls_summary.get('avg_brier') is not None:
                print(f"  平均Brier: {cls_summary['avg_brier']:.4f}")
            if cls_summary.get('avg_ece') is not None:
                print(f"  平均ECE: {cls_summary['avg_ece']:.4f}")
    
    if 'regression' in results['summary']:
        reg_summary = results['summary']['regression']
        print(f"回归任务摘要 ({reg_summary.get('n_tasks_evaluated', 0)} 个任务):")
        if reg_summary.get('avg_r2') is not None:
            print(f"  平均R²: {reg_summary['avg_r2']:.4f}")
        if reg_summary.get('avg_rmse') is not None:
            print(f"  平均RMSE: {reg_summary['avg_rmse']:.4f}")
        if reg_summary.get('avg_mae') is not None:
            print(f"  平均MAE: {reg_summary['avg_mae']:.4f}")

    # 保存可视化
    if args.save_plots and args.with_uncertainty:
        try:
            cal_curves = results.get('calibration_curves', {})
            if cal_curves:
                # Keep order consistent with CLASSIFICATION_TASKS
                tasks = [t for t in CLASSIFICATION_TASKS if t in cal_curves]
                n = len(tasks)
                cols = 4
                rows = int(np.ceil(n / cols)) if n > 0 else 1
                fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.8 * rows))
                axes = np.array(axes).reshape(-1)
                for i, t in enumerate(tasks):
                    bc = np.array(cal_curves[t]['bin_centers'])
                    fp = np.array(cal_curves[t]['frac_pos'])
                    plot_reliability_diagram(axes[i], bc, fp, title=f'{t}')
                for j in range(n, rows * cols):
                    axes[j].axis('off')
                fig.tight_layout()
                (ckpt_dir / f'calibration_curves_{args.split}.png').write_bytes(fig_to_png_bytes(fig))
                plt.close(fig)
        except Exception as e:
            logger.warning(f'Failed to save calibration plots: {e}')

    # Save calibration curves as CSV for reproducibility
    if args.with_uncertainty and results.get('calibration_curves'):
        try:
            import pandas as pd
            rows = []
            for ep, data in results['calibration_curves'].items():
                bc = data['bin_centers']
                fp = data['frac_pos']
                ct = data['counts']
                for x, y, c in zip(bc, fp, ct):
                    rows.append({'endpoint': ep, 'bin_center': x, 'frac_pos': y, 'count': int(c)})
            df = pd.DataFrame(rows)
            df.to_csv(ckpt_dir / f'calibration_curves_{args.split}.csv', index=False)
        except Exception as e:
            logger.warning(f'Failed to save calibration CSV: {e}')

    if args.save_plots and args.with_ad and ad_report is not None:
        try:
            scores = np.array(ad_report['scores'])
            thr = ad_report['threshold']
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.hist(scores, bins=30, alpha=0.7)
            ax.axvline(thr, color='red', linestyle='--', label=f'Threshold={thr:.2f}')
            ax.set_title('Embedding AD Mahalanobis distances')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Count')
            ax.legend()
            fig.tight_layout()
            (ckpt_dir / f'ad_mahalanobis_{args.split}.png').write_bytes(fig_to_png_bytes(fig))
            plt.close(fig)
        except Exception as e:
            logger.warning(f'Failed to save AD plots: {e}')

def fig_to_png_bytes(fig) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

if __name__ == '__main__':
    main()
