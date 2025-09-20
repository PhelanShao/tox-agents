#!/usr/bin/env python3
"""
A2 不确定性量化分析运行脚本
针对三个已训练模型进行完整的不确定性和适用域分析
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class A2AnalysisRunner:
    """A2分析运行器"""
    
    def __init__(self):
        self.experiment_paths = [
            "/mnt/backup2/ai4s/backupunimolpy/responds-work/ToxD4C/experiments/toxd4c_baseline_complete_20250903_220804",
            "/mnt/backup2/ai4s/backupunimolpy/responds-work/ToxD4C/experiments/toxd4c_baseline_complete_20250903_223754", 
            "/mnt/backup2/ai4s/backupunimolpy/responds-work/ToxD4C/experiments/toxd4c_baseline_complete_20250904_094335"
        ]
        
        self.output_dir = Path("a2_uncertainty_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # ToxD4C终点
        self.classification_endpoints = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        self.regression_endpoints = [
            'IGC50', 'LC50', 'LC50DM', 'LLNA', 'LOAEL'
        ]
    
    def check_experiment_files(self) -> bool:
        """检查实验文件是否存在"""
        logger.info("🔍 Checking experiment files...")
        
        all_exist = True
        for exp_path in self.experiment_paths:
            exp_dir = Path(exp_path)
            if not exp_dir.exists():
                logger.error(f"Experiment directory not found: {exp_path}")
                all_exist = False
                continue
            
            # 检查关键文件
            required_files = ['config.json']
            optional_files = ['test_metrics.json', 'test_predictions.json', 'test_results.json']
            
            for file_name in required_files:
                if not (exp_dir / file_name).exists():
                    logger.warning(f"Required file missing: {exp_path}/{file_name}")
            
            found_optional = False
            for file_name in optional_files:
                if (exp_dir / file_name).exists():
                    found_optional = True
                    logger.info(f"Found: {exp_path}/{file_name}")
            
            if not found_optional:
                logger.warning(f"No prediction/result files found in: {exp_path}")
        
        return all_exist
    
    def extract_metrics_from_experiments(self) -> Dict:
        """从实验中提取指标"""
        logger.info("📊 Extracting metrics from experiments...")

        all_metrics = {
            'classification': {},
            'regression': {}
        }

        for i, exp_path in enumerate(self.experiment_paths):
            exp_dir = Path(exp_path)

            # 查找结果文件 (在checkpoints目录中)
            possible_files = [
                exp_dir / 'test_metrics.json',
                exp_dir / 'checkpoints' / 'test_metrics.json',
                exp_dir / 'checkpoints' / f'{exp_dir.name}_results.json',
                exp_dir / 'checkpoints' / 'toxd4c_baseline_complete_results.json'
            ]

            metrics = None
            for metrics_file in possible_files:
                if metrics_file.exists():
                    logger.info(f"Loading metrics from: {metrics_file}")
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    break

            if metrics is None:
                logger.warning(f"No metrics file found for experiment: {exp_path}")
                continue

            # 解析指标 - 检查是否有final_metrics字段
            if 'final_metrics' in metrics:
                final_metrics = metrics['final_metrics']

                # 分类指标 (前12个任务是分类)
                for j, endpoint in enumerate(self.classification_endpoints):
                    auc_key = f'task_{j}_auc'
                    acc_key = f'task_{j}_accuracy'

                    if auc_key in final_metrics:
                        if endpoint not in all_metrics['classification']:
                            all_metrics['classification'][endpoint] = {
                                'auc': [], 'accuracy': []
                            }

                        all_metrics['classification'][endpoint]['auc'].append(
                            final_metrics[auc_key]
                        )
                        all_metrics['classification'][endpoint]['accuracy'].append(
                            final_metrics.get(acc_key, 0)
                        )

                # 回归指标 (任务12-16是回归，对应task_0-task_4的回归指标)
                for j, endpoint in enumerate(self.regression_endpoints):
                    r2_key = f'task_{j}_r2'
                    rmse_key = f'task_{j}_rmse'
                    mse_key = f'task_{j}_mse'

                    if r2_key in final_metrics:
                        if endpoint not in all_metrics['regression']:
                            all_metrics['regression'][endpoint] = {
                                'r2': [], 'rmse': [], 'mse': []
                            }

                        all_metrics['regression'][endpoint]['r2'].append(
                            final_metrics[r2_key]
                        )
                        all_metrics['regression'][endpoint]['rmse'].append(
                            final_metrics.get(rmse_key, 0)
                        )
                        all_metrics['regression'][endpoint]['mse'].append(
                            final_metrics.get(mse_key, 0)
                        )

            # 兼容旧格式
            elif 'classification' in metrics and 'regression' in metrics:
                # 原有的解析逻辑保持不变
                for j, endpoint in enumerate(self.classification_endpoints):
                    if str(j) in metrics['classification'].get('auc', {}):
                        if endpoint not in all_metrics['classification']:
                            all_metrics['classification'][endpoint] = {
                                'auc': [], 'accuracy': []
                            }

                        all_metrics['classification'][endpoint]['auc'].append(
                            metrics['classification']['auc'][str(j)]
                        )
                        all_metrics['classification'][endpoint]['accuracy'].append(
                            metrics['classification'].get('accuracy', {}).get(str(j), 0)
                        )
        
        return all_metrics
    
    def compute_ensemble_statistics(self, metrics: Dict) -> Dict:
        """计算集成统计"""
        logger.info("📈 Computing ensemble statistics...")
        
        ensemble_stats = {
            'classification': {},
            'regression': {}
        }
        
        # 分类统计
        for endpoint, endpoint_metrics in metrics['classification'].items():
            stats = {}
            for metric_name, values in endpoint_metrics.items():
                if len(values) > 0:
                    stats[f'{metric_name}_mean'] = np.mean(values)
                    stats[f'{metric_name}_std'] = np.std(values)
                    stats[f'{metric_name}_min'] = np.min(values)
                    stats[f'{metric_name}_max'] = np.max(values)
            
            ensemble_stats['classification'][endpoint] = stats
        
        # 回归统计
        for endpoint, endpoint_metrics in metrics['regression'].items():
            stats = {}
            for metric_name, values in endpoint_metrics.items():
                if len(values) > 0:
                    stats[f'{metric_name}_mean'] = np.mean(values)
                    stats[f'{metric_name}_std'] = np.std(values)
                    stats[f'{metric_name}_min'] = np.min(values)
                    stats[f'{metric_name}_max'] = np.max(values)
            
            ensemble_stats['regression'][endpoint] = stats
        
        return ensemble_stats
    
    def create_uncertainty_visualization(self, ensemble_stats: Dict):
        """创建不确定性可视化"""
        logger.info("🎨 Creating uncertainty visualizations...")
        
        # 1. 分类任务不确定性图
        if ensemble_stats['classification']:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Classification Uncertainty Analysis', fontsize=16, fontweight='bold')
            
            endpoints = list(ensemble_stats['classification'].keys())
            
            # AUC均值和标准差
            auc_means = [ensemble_stats['classification'][ep].get('auc_mean', 0) for ep in endpoints]
            auc_stds = [ensemble_stats['classification'][ep].get('auc_std', 0) for ep in endpoints]
            
            ax1 = axes[0, 0]
            bars = ax1.bar(range(len(endpoints)), auc_means, yerr=auc_stds, capsize=5, alpha=0.7)
            ax1.set_xlabel('Endpoints')
            ax1.set_ylabel('AUC')
            ax1.set_title('AUC with Uncertainty (Mean ± Std)')
            ax1.set_xticks(range(len(endpoints)))
            ax1.set_xticklabels(endpoints, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # PR-AUC均值和标准差
            pr_auc_means = [ensemble_stats['classification'][ep].get('pr_auc_mean', 0) for ep in endpoints]
            pr_auc_stds = [ensemble_stats['classification'][ep].get('pr_auc_std', 0) for ep in endpoints]
            
            ax2 = axes[0, 1]
            ax2.bar(range(len(endpoints)), pr_auc_means, yerr=pr_auc_stds, capsize=5, alpha=0.7, color='orange')
            ax2.set_xlabel('Endpoints')
            ax2.set_ylabel('PR-AUC')
            ax2.set_title('PR-AUC with Uncertainty (Mean ± Std)')
            ax2.set_xticks(range(len(endpoints)))
            ax2.set_xticklabels(endpoints, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # 不确定性分布
            ax3 = axes[1, 0]
            ax3.hist(auc_stds, bins=10, alpha=0.7, color='green')
            ax3.set_xlabel('AUC Standard Deviation')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of AUC Uncertainty')
            ax3.grid(True, alpha=0.3)
            
            # 性能vs不确定性散点图
            ax4 = axes[1, 1]
            ax4.scatter(auc_means, auc_stds, alpha=0.7, s=100)
            ax4.set_xlabel('AUC Mean')
            ax4.set_ylabel('AUC Standard Deviation')
            ax4.set_title('Performance vs Uncertainty')
            ax4.grid(True, alpha=0.3)
            
            # 添加终点标签
            for i, endpoint in enumerate(endpoints):
                ax4.annotate(endpoint, (auc_means[i], auc_stds[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'classification_uncertainty.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. 回归任务不确定性图
        if ensemble_stats['regression']:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Regression Uncertainty Analysis', fontsize=16, fontweight='bold')
            
            endpoints = list(ensemble_stats['regression'].keys())
            
            # R²均值和标准差
            r2_means = [ensemble_stats['regression'][ep].get('r2_mean', 0) for ep in endpoints]
            r2_stds = [ensemble_stats['regression'][ep].get('r2_std', 0) for ep in endpoints]
            
            ax1 = axes[0, 0]
            ax1.bar(range(len(endpoints)), r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
            ax1.set_xlabel('Endpoints')
            ax1.set_ylabel('R²')
            ax1.set_title('R² with Uncertainty (Mean ± Std)')
            ax1.set_xticks(range(len(endpoints)))
            ax1.set_xticklabels(endpoints, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # RMSE均值和标准差
            rmse_means = [ensemble_stats['regression'][ep].get('rmse_mean', 0) for ep in endpoints]
            rmse_stds = [ensemble_stats['regression'][ep].get('rmse_std', 0) for ep in endpoints]
            
            ax2 = axes[0, 1]
            ax2.bar(range(len(endpoints)), rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color='red')
            ax2.set_xlabel('Endpoints')
            ax2.set_ylabel('RMSE')
            ax2.set_title('RMSE with Uncertainty (Mean ± Std)')
            ax2.set_xticks(range(len(endpoints)))
            ax2.set_xticklabels(endpoints, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # 不确定性分布
            ax3 = axes[1, 0]
            ax3.hist(r2_stds, bins=10, alpha=0.7, color='purple')
            ax3.set_xlabel('R² Standard Deviation')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of R² Uncertainty')
            ax3.grid(True, alpha=0.3)
            
            # 性能vs不确定性散点图
            ax4 = axes[1, 1]
            ax4.scatter(r2_means, r2_stds, alpha=0.7, s=100, color='purple')
            ax4.set_xlabel('R² Mean')
            ax4.set_ylabel('R² Standard Deviation')
            ax4.set_title('Performance vs Uncertainty')
            ax4.grid(True, alpha=0.3)
            
            # 添加终点标签
            for i, endpoint in enumerate(endpoints):
                ax4.annotate(endpoint, (r2_means[i], r2_stds[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'regression_uncertainty.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_summary_report(self, ensemble_stats: Dict):
        """生成摘要报告"""
        logger.info("📋 Generating summary report...")
        
        # 创建摘要表格
        summary_data = []
        
        # 分类任务摘要
        for endpoint, stats in ensemble_stats['classification'].items():
            summary_data.append({
                'Task': 'Classification',
                'Endpoint': endpoint,
                'Metric': 'AUC',
                'Mean': stats.get('auc_mean', 0),
                'Std': stats.get('auc_std', 0),
                'Min': stats.get('auc_min', 0),
                'Max': stats.get('auc_max', 0)
            })
        
        # 回归任务摘要
        for endpoint, stats in ensemble_stats['regression'].items():
            summary_data.append({
                'Task': 'Regression',
                'Endpoint': endpoint,
                'Metric': 'R²',
                'Mean': stats.get('r2_mean', 0),
                'Std': stats.get('r2_std', 0),
                'Min': stats.get('r2_min', 0),
                'Max': stats.get('r2_max', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存摘要表格
        summary_file = self.output_dir / 'uncertainty_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to: {summary_file}")
        
        # 打印摘要统计
        print("\n" + "="*80)
        print("A2 UNCERTAINTY QUANTIFICATION SUMMARY")
        print("="*80)
        
        print(f"\nAnalyzed {len(self.experiment_paths)} models:")
        for i, path in enumerate(self.experiment_paths):
            print(f"  {i+1}. {Path(path).name}")
        
        print(f"\nClassification Endpoints: {len(ensemble_stats['classification'])}")
        if ensemble_stats['classification']:
            avg_auc_uncertainty = np.mean([stats.get('auc_std', 0) for stats in ensemble_stats['classification'].values()])
            print(f"  Average AUC uncertainty (std): {avg_auc_uncertainty:.3f}")
        
        print(f"\nRegression Endpoints: {len(ensemble_stats['regression'])}")
        if ensemble_stats['regression']:
            avg_r2_uncertainty = np.mean([stats.get('r2_std', 0) for stats in ensemble_stats['regression'].values()])
            print(f"  Average R² uncertainty (std): {avg_r2_uncertainty:.3f}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("Generated files:")
        print("  - uncertainty_summary.csv")
        print("  - classification_uncertainty.png")
        print("  - regression_uncertainty.png")
    
    def run_analysis(self):
        """运行完整的A2分析"""
        logger.info("🚀 Starting A2 uncertainty quantification analysis...")
        
        # 1. 检查文件
        if not self.check_experiment_files():
            logger.error("Some experiment files are missing. Please check the paths.")
            return
        
        # 2. 提取指标
        metrics = self.extract_metrics_from_experiments()
        
        # 3. 计算集成统计
        ensemble_stats = self.compute_ensemble_statistics(metrics)
        
        # 4. 创建可视化
        self.create_uncertainty_visualization(ensemble_stats)
        
        # 5. 生成报告
        self.generate_summary_report(ensemble_stats)
        
        # 6. 保存完整结果
        results = {
            'experiment_paths': self.experiment_paths,
            'raw_metrics': metrics,
            'ensemble_statistics': ensemble_stats
        }
        
        results_file = self.output_dir / 'a2_complete_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("✅ A2 analysis completed successfully!")

def main():
    """主函数"""
    runner = A2AnalysisRunner()
    runner.run_analysis()

if __name__ == "__main__":
    main()
