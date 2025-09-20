#!/usr/bin/env python3
"""
随机 vs Scaffold 切分性能对比分析
生成A1任务所需的对比图表
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置绘图样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SplitComparisonAnalyzer:
    """切分方法性能对比分析器"""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # ToxD4C终点信息
        self.classification_endpoints = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        self.regression_endpoints = [
            'IGC50', 'LC50', 'LC50DM', 'LLNA', 'LOAEL'
        ]
    
    def load_experiment_results(self, experiment_path: str) -> Dict:
        """加载实验结果"""
        exp_path = Path(experiment_path)
        
        # 查找结果文件
        results_files = {
            'metrics': exp_path / 'test_metrics.json',
            'config': exp_path / 'config.json',
            'predictions': exp_path / 'test_predictions.json'
        }
        
        results = {}
        for key, file_path in results_files.items():
            if file_path.exists():
                with open(file_path, 'r') as f:
                    results[key] = json.load(f)
            else:
                logger.warning(f"File not found: {file_path}")
        
        return results
    
    def extract_metrics_by_endpoint(self, results: Dict) -> pd.DataFrame:
        """提取每个终点的指标"""
        metrics_data = []
        
        if 'metrics' not in results:
            logger.error("No metrics found in results")
            return pd.DataFrame()
        
        metrics = results['metrics']
        
        # 分类终点
        for i, endpoint in enumerate(self.classification_endpoints):
            if 'classification' in metrics:
                cls_metrics = metrics['classification']
                row = {
                    'endpoint': endpoint,
                    'task_type': 'classification',
                    'endpoint_idx': i,
                    'auc': cls_metrics.get('auc', {}).get(str(i), np.nan),
                    'pr_auc': cls_metrics.get('pr_auc', {}).get(str(i), np.nan),
                    'f1': cls_metrics.get('f1', {}).get(str(i), np.nan),
                    'accuracy': cls_metrics.get('accuracy', {}).get(str(i), np.nan),
                    'precision': cls_metrics.get('precision', {}).get(str(i), np.nan),
                    'recall': cls_metrics.get('recall', {}).get(str(i), np.nan)
                }
                metrics_data.append(row)
        
        # 回归终点
        for i, endpoint in enumerate(self.regression_endpoints):
            if 'regression' in metrics:
                reg_metrics = metrics['regression']
                row = {
                    'endpoint': endpoint,
                    'task_type': 'regression',
                    'endpoint_idx': i,
                    'r2': reg_metrics.get('r2', {}).get(str(i), np.nan),
                    'rmse': reg_metrics.get('rmse', {}).get(str(i), np.nan),
                    'mae': reg_metrics.get('mae', {}).get(str(i), np.nan),
                    'pearson': reg_metrics.get('pearson', {}).get(str(i), np.nan),
                    'spearman': reg_metrics.get('spearman', {}).get(str(i), np.nan)
                }
                metrics_data.append(row)
        
        return pd.DataFrame(metrics_data)
    
    def compare_split_methods(self, random_results: Dict, scaffold_results: Dict) -> pd.DataFrame:
        """对比两种切分方法的结果"""
        
        # 提取指标
        random_df = self.extract_metrics_by_endpoint(random_results)
        scaffold_df = self.extract_metrics_by_endpoint(scaffold_results)
        
        if random_df.empty or scaffold_df.empty:
            logger.error("Failed to extract metrics from results")
            return pd.DataFrame()
        
        # 添加切分方法标识
        random_df['split_method'] = 'Random'
        scaffold_df['split_method'] = 'Scaffold'
        
        # 合并数据
        combined_df = pd.concat([random_df, scaffold_df], ignore_index=True)
        
        return combined_df
    
    def create_performance_comparison_plot(self, df: pd.DataFrame, save_path: str = None):
        """创建性能对比图"""
        
        if df.empty:
            logger.error("No data to plot")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Random vs Scaffold Split Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. 分类任务AUC对比
        cls_data = df[df['task_type'] == 'classification'].copy()
        if not cls_data.empty:
            ax1 = axes[0, 0]
            pivot_auc = cls_data.pivot(index='endpoint', columns='split_method', values='auc')
            
            x = np.arange(len(pivot_auc.index))
            width = 0.35
            
            ax1.bar(x - width/2, pivot_auc['Random'], width, label='Random', alpha=0.8)
            ax1.bar(x + width/2, pivot_auc['Scaffold'], width, label='Scaffold', alpha=0.8)
            
            ax1.set_xlabel('Classification Endpoints')
            ax1.set_ylabel('AUC')
            ax1.set_title('Classification Performance (AUC)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(pivot_auc.index, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 分类任务PR-AUC对比
        if not cls_data.empty:
            ax2 = axes[0, 1]
            pivot_pr = cls_data.pivot(index='endpoint', columns='split_method', values='pr_auc')
            
            x = np.arange(len(pivot_pr.index))
            ax2.bar(x - width/2, pivot_pr['Random'], width, label='Random', alpha=0.8)
            ax2.bar(x + width/2, pivot_pr['Scaffold'], width, label='Scaffold', alpha=0.8)
            
            ax2.set_xlabel('Classification Endpoints')
            ax2.set_ylabel('PR-AUC')
            ax2.set_title('Classification Performance (PR-AUC)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(pivot_pr.index, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 回归任务R²对比
        reg_data = df[df['task_type'] == 'regression'].copy()
        if not reg_data.empty:
            ax3 = axes[1, 0]
            pivot_r2 = reg_data.pivot(index='endpoint', columns='split_method', values='r2')
            
            x = np.arange(len(pivot_r2.index))
            ax3.bar(x - width/2, pivot_r2['Random'], width, label='Random', alpha=0.8)
            ax3.bar(x + width/2, pivot_r2['Scaffold'], width, label='Scaffold', alpha=0.8)
            
            ax3.set_xlabel('Regression Endpoints')
            ax3.set_ylabel('R²')
            ax3.set_title('Regression Performance (R²)')
            ax3.set_xticks(x)
            ax3.set_xticklabels(pivot_r2.index, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 性能差异散点图
        ax4 = axes[1, 1]
        
        # 计算性能差异
        for task_type in ['classification', 'regression']:
            task_data = df[df['task_type'] == task_type].copy()
            if task_data.empty:
                continue
                
            metric = 'auc' if task_type == 'classification' else 'r2'
            pivot = task_data.pivot(index='endpoint', columns='split_method', values=metric)
            
            if 'Random' in pivot.columns and 'Scaffold' in pivot.columns:
                diff = pivot['Random'] - pivot['Scaffold']
                
                color = 'blue' if task_type == 'classification' else 'red'
                marker = 'o' if task_type == 'classification' else 's'
                
                ax4.scatter(pivot['Random'], pivot['Scaffold'], 
                           c=color, marker=marker, alpha=0.7, s=100,
                           label=f'{task_type.title()} ({metric.upper()})')
        
        # 添加对角线
        lims = [
            np.min([ax4.get_xlim(), ax4.get_ylim()]),
            np.max([ax4.get_xlim(), ax4.get_ylim()]),
        ]
        ax4.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        ax4.set_xlabel('Random Split Performance')
        ax4.set_ylabel('Scaffold Split Performance')
        ax4.set_title('Performance Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance comparison plot saved to: {save_path}")
        
        plt.show()
    
    def create_ladder_plot(self, df: pd.DataFrame, save_path: str = None):
        """创建梯形图(Ladder Plot)显示性能变化"""
        
        if df.empty:
            logger.error("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Performance Changes: Random → Scaffold Split', fontsize=16, fontweight='bold')
        
        # 分类任务梯形图
        cls_data = df[df['task_type'] == 'classification'].copy()
        if not cls_data.empty:
            pivot_auc = cls_data.pivot(index='endpoint', columns='split_method', values='auc')
            
            if 'Random' in pivot_auc.columns and 'Scaffold' in pivot_auc.columns:
                for i, endpoint in enumerate(pivot_auc.index):
                    random_val = pivot_auc.loc[endpoint, 'Random']
                    scaffold_val = pivot_auc.loc[endpoint, 'Scaffold']
                    
                    if pd.notna(random_val) and pd.notna(scaffold_val):
                        # 连线
                        ax1.plot([0, 1], [random_val, scaffold_val], 'o-', 
                                alpha=0.7, linewidth=2, markersize=8)
                        
                        # 标注终点名称
                        ax1.text(-0.05, random_val, endpoint, ha='right', va='center', fontsize=9)
                
                ax1.set_xlim(-0.3, 1.3)
                ax1.set_xticks([0, 1])
                ax1.set_xticklabels(['Random', 'Scaffold'])
                ax1.set_ylabel('AUC')
                ax1.set_title('Classification Performance Changes')
                ax1.grid(True, alpha=0.3)
        
        # 回归任务梯形图
        reg_data = df[df['task_type'] == 'regression'].copy()
        if not reg_data.empty:
            pivot_r2 = reg_data.pivot(index='endpoint', columns='split_method', values='r2')
            
            if 'Random' in pivot_r2.columns and 'Scaffold' in pivot_r2.columns:
                for i, endpoint in enumerate(pivot_r2.index):
                    random_val = pivot_r2.loc[endpoint, 'Random']
                    scaffold_val = pivot_r2.loc[endpoint, 'Scaffold']
                    
                    if pd.notna(random_val) and pd.notna(scaffold_val):
                        # 连线
                        ax2.plot([0, 1], [random_val, scaffold_val], 's-', 
                                alpha=0.7, linewidth=2, markersize=8)
                        
                        # 标注终点名称
                        ax2.text(-0.05, random_val, endpoint, ha='right', va='center', fontsize=9)
                
                ax2.set_xlim(-0.3, 1.3)
                ax2.set_xticks([0, 1])
                ax2.set_xticklabels(['Random', 'Scaffold'])
                ax2.set_ylabel('R²')
                ax2.set_title('Regression Performance Changes')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ladder plot saved to: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare Random vs Scaffold split performance')
    parser.add_argument('--random_results', type=str, required=True,
                       help='Path to random split experiment results')
    parser.add_argument('--scaffold_results', type=str, required=True,
                       help='Path to scaffold split experiment results')
    parser.add_argument('--output_dir', type=str, default='split_comparison_results',
                       help='Output directory for plots and tables')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = SplitComparisonAnalyzer(args.output_dir)
    
    # 加载结果
    logger.info("Loading experiment results...")
    random_results = analyzer.load_experiment_results(args.random_results)
    scaffold_results = analyzer.load_experiment_results(args.scaffold_results)
    
    # 对比分析
    logger.info("Comparing split methods...")
    comparison_df = analyzer.compare_split_methods(random_results, scaffold_results)
    
    if comparison_df.empty:
        logger.error("Failed to create comparison data")
        return
    
    # 保存对比表格
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    comparison_df.to_csv(output_dir / 'split_comparison_metrics.csv', index=False)
    logger.info(f"Comparison metrics saved to: {output_dir / 'split_comparison_metrics.csv'}")
    
    # 生成图表
    logger.info("Generating comparison plots...")
    
    # 性能对比图
    analyzer.create_performance_comparison_plot(
        comparison_df, 
        save_path=output_dir / 'performance_comparison.png'
    )
    
    # 梯形图
    analyzer.create_ladder_plot(
        comparison_df,
        save_path=output_dir / 'ladder_plot.png'
    )
    
    # 生成统计摘要
    logger.info("Generating statistical summary...")
    
    summary_stats = []
    for task_type in ['classification', 'regression']:
        task_data = comparison_df[comparison_df['task_type'] == task_type]
        if task_data.empty:
            continue
            
        metric = 'auc' if task_type == 'classification' else 'r2'
        
        random_vals = task_data[task_data['split_method'] == 'Random'][metric].dropna()
        scaffold_vals = task_data[task_data['split_method'] == 'Scaffold'][metric].dropna()
        
        if len(random_vals) > 0 and len(scaffold_vals) > 0:
            summary_stats.append({
                'task_type': task_type,
                'metric': metric,
                'random_mean': random_vals.mean(),
                'random_std': random_vals.std(),
                'scaffold_mean': scaffold_vals.mean(),
                'scaffold_std': scaffold_vals.std(),
                'mean_difference': random_vals.mean() - scaffold_vals.mean(),
                'n_endpoints': len(random_vals)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / 'split_comparison_summary.csv', index=False)
    
    logger.info("Analysis complete!")
    print("\n" + "="*50)
    print("SPLIT COMPARISON SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
