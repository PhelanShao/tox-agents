#!/usr/bin/env python3
"""
A2 不确定性量化分析 - 简化版本
基于三个已训练模型的结果进行深度集成不确定性分析
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置绘图样式
plt.style.use('default')
sns.set_palette("husl")

class SimpleUncertaintyAnalyzer:
    """简化的不确定性分析器"""
    
    def __init__(self):
        self.experiment_paths = [
            "experiments/toxd4c_baseline_complete_20250903_220804",
            "experiments/toxd4c_baseline_complete_20250903_223754", 
            "experiments/toxd4c_baseline_complete_20250904_094335"
        ]
        
        self.output_dir = Path("a2_uncertainty_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # ToxD4C终点名称
        self.classification_endpoints = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        self.regression_endpoints = [
            'IGC50', 'LC50', 'LC50DM', 'LLNA', 'LOAEL'
        ]
    
    def load_all_metrics(self):
        """加载所有实验的指标"""
        logger.info("📊 Loading metrics from all experiments...")
        
        all_data = []
        
        for exp_path in self.experiment_paths:
            results_file = Path(exp_path) / "checkpoints" / "toxd4c_baseline_complete_results.json"
            
            if not results_file.exists():
                logger.warning(f"Results file not found: {results_file}")
                continue
            
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            if 'final_metrics' not in data:
                logger.warning(f"No final_metrics in {exp_path}")
                continue
            
            metrics = data['final_metrics']
            exp_data = {'experiment': Path(exp_path).name}
            
            # 分类指标
            for i, endpoint in enumerate(self.classification_endpoints):
                auc_key = f'task_{i}_auc'
                acc_key = f'task_{i}_accuracy'
                
                if auc_key in metrics:
                    exp_data[f'{endpoint}_auc'] = metrics[auc_key]
                    exp_data[f'{endpoint}_accuracy'] = metrics.get(acc_key, 0)
            
            # 回归指标
            for i, endpoint in enumerate(self.regression_endpoints):
                r2_key = f'task_{i}_r2'
                rmse_key = f'task_{i}_rmse'
                
                if r2_key in metrics:
                    exp_data[f'{endpoint}_r2'] = metrics[r2_key]
                    exp_data[f'{endpoint}_rmse'] = metrics.get(rmse_key, 0)
            
            all_data.append(exp_data)
        
        return pd.DataFrame(all_data)
    
    def compute_uncertainty_statistics(self, df):
        """计算不确定性统计"""
        logger.info("📈 Computing uncertainty statistics...")
        
        uncertainty_stats = {
            'classification': {},
            'regression': {}
        }
        
        # 分类任务不确定性
        for endpoint in self.classification_endpoints:
            auc_col = f'{endpoint}_auc'
            acc_col = f'{endpoint}_accuracy'
            
            if auc_col in df.columns:
                auc_values = df[auc_col].dropna()
                acc_values = df[acc_col].dropna()
                
                if len(auc_values) > 1:
                    uncertainty_stats['classification'][endpoint] = {
                        'auc_mean': auc_values.mean(),
                        'auc_std': auc_values.std(),
                        'auc_min': auc_values.min(),
                        'auc_max': auc_values.max(),
                        'auc_values': auc_values.tolist(),
                        'accuracy_mean': acc_values.mean(),
                        'accuracy_std': acc_values.std(),
                        'accuracy_values': acc_values.tolist(),
                        'n_models': len(auc_values)
                    }
        
        # 回归任务不确定性
        for endpoint in self.regression_endpoints:
            r2_col = f'{endpoint}_r2'
            rmse_col = f'{endpoint}_rmse'
            
            if r2_col in df.columns:
                r2_values = df[r2_col].dropna()
                rmse_values = df[rmse_col].dropna()
                
                if len(r2_values) > 1:
                    uncertainty_stats['regression'][endpoint] = {
                        'r2_mean': r2_values.mean(),
                        'r2_std': r2_values.std(),
                        'r2_min': r2_values.min(),
                        'r2_max': r2_values.max(),
                        'r2_values': r2_values.tolist(),
                        'rmse_mean': rmse_values.mean(),
                        'rmse_std': rmse_values.std(),
                        'rmse_values': rmse_values.tolist(),
                        'n_models': len(r2_values)
                    }
        
        return uncertainty_stats
    
    def create_uncertainty_plots(self, uncertainty_stats):
        """创建不确定性可视化"""
        logger.info("🎨 Creating uncertainty visualizations...")
        
        # 1. 分类任务不确定性
        if uncertainty_stats['classification']:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Classification Uncertainty Analysis (Deep Ensemble)', fontsize=16, fontweight='bold')
            
            endpoints = list(uncertainty_stats['classification'].keys())
            auc_means = [uncertainty_stats['classification'][ep]['auc_mean'] for ep in endpoints]
            auc_stds = [uncertainty_stats['classification'][ep]['auc_std'] for ep in endpoints]
            
            # AUC均值和不确定性
            ax1 = axes[0, 0]
            bars = ax1.bar(range(len(endpoints)), auc_means, yerr=auc_stds, capsize=5, alpha=0.7)
            ax1.set_xlabel('Endpoints')
            ax1.set_ylabel('AUC')
            ax1.set_title('AUC with Epistemic Uncertainty (Mean ± Std)')
            ax1.set_xticks(range(len(endpoints)))
            ax1.set_xticklabels(endpoints, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # 不确定性分布
            ax2 = axes[0, 1]
            ax2.hist(auc_stds, bins=8, alpha=0.7, color='orange')
            ax2.set_xlabel('AUC Standard Deviation')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Epistemic Uncertainty')
            ax2.grid(True, alpha=0.3)
            
            # 性能vs不确定性
            ax3 = axes[1, 0]
            scatter = ax3.scatter(auc_means, auc_stds, alpha=0.7, s=100, c=range(len(endpoints)), cmap='viridis')
            ax3.set_xlabel('AUC Mean')
            ax3.set_ylabel('AUC Standard Deviation')
            ax3.set_title('Performance vs Uncertainty Trade-off')
            ax3.grid(True, alpha=0.3)
            
            # 添加终点标签
            for i, endpoint in enumerate(endpoints):
                ax3.annotate(endpoint, (auc_means[i], auc_stds[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 模型间变异性
            ax4 = axes[1, 1]
            all_values = []
            labels = []
            for endpoint in endpoints[:6]:  # 只显示前6个以避免拥挤
                values = uncertainty_stats['classification'][endpoint]['auc_values']
                all_values.extend(values)
                labels.extend([endpoint] * len(values))
            
            if all_values:
                df_plot = pd.DataFrame({'AUC': all_values, 'Endpoint': labels})
                sns.boxplot(data=df_plot, x='Endpoint', y='AUC', ax=ax4)
                ax4.set_title('Model Variability (First 6 Endpoints)')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'classification_uncertainty.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. 回归任务不确定性
        if uncertainty_stats['regression']:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Regression Uncertainty Analysis (Deep Ensemble)', fontsize=16, fontweight='bold')
            
            endpoints = list(uncertainty_stats['regression'].keys())
            r2_means = [uncertainty_stats['regression'][ep]['r2_mean'] for ep in endpoints]
            r2_stds = [uncertainty_stats['regression'][ep]['r2_std'] for ep in endpoints]
            
            # R²均值和不确定性
            ax1 = axes[0, 0]
            bars = ax1.bar(range(len(endpoints)), r2_means, yerr=r2_stds, capsize=5, alpha=0.7, color='green')
            ax1.set_xlabel('Endpoints')
            ax1.set_ylabel('R²')
            ax1.set_title('R² with Epistemic Uncertainty (Mean ± Std)')
            ax1.set_xticks(range(len(endpoints)))
            ax1.set_xticklabels(endpoints, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # 不确定性分布
            ax2 = axes[0, 1]
            ax2.hist(r2_stds, bins=5, alpha=0.7, color='red')
            ax2.set_xlabel('R² Standard Deviation')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Epistemic Uncertainty')
            ax2.grid(True, alpha=0.3)
            
            # 性能vs不确定性
            ax3 = axes[1, 0]
            scatter = ax3.scatter(r2_means, r2_stds, alpha=0.7, s=100, c=range(len(endpoints)), cmap='plasma')
            ax3.set_xlabel('R² Mean')
            ax3.set_ylabel('R² Standard Deviation')
            ax3.set_title('Performance vs Uncertainty Trade-off')
            ax3.grid(True, alpha=0.3)
            
            # 添加终点标签
            for i, endpoint in enumerate(endpoints):
                ax3.annotate(endpoint, (r2_means[i], r2_stds[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # 模型间变异性
            ax4 = axes[1, 1]
            all_values = []
            labels = []
            for endpoint in endpoints:
                values = uncertainty_stats['regression'][endpoint]['r2_values']
                all_values.extend(values)
                labels.extend([endpoint] * len(values))
            
            if all_values:
                df_plot = pd.DataFrame({'R²': all_values, 'Endpoint': labels})
                sns.boxplot(data=df_plot, x='Endpoint', y='R²', ax=ax4)
                ax4.set_title('Model Variability')
                ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'regression_uncertainty.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_summary_table(self, uncertainty_stats):
        """生成摘要表格"""
        logger.info("📋 Generating summary table...")
        
        summary_data = []
        
        # 分类任务
        for endpoint, stats in uncertainty_stats['classification'].items():
            summary_data.append({
                'Task_Type': 'Classification',
                'Endpoint': endpoint,
                'Primary_Metric': 'AUC',
                'Mean': f"{stats['auc_mean']:.3f}",
                'Std': f"{stats['auc_std']:.3f}",
                'Min': f"{stats['auc_min']:.3f}",
                'Max': f"{stats['auc_max']:.3f}",
                'N_Models': stats['n_models'],
                'Uncertainty_Level': 'High' if stats['auc_std'] > 0.05 else 'Medium' if stats['auc_std'] > 0.02 else 'Low'
            })
        
        # 回归任务
        for endpoint, stats in uncertainty_stats['regression'].items():
            summary_data.append({
                'Task_Type': 'Regression',
                'Endpoint': endpoint,
                'Primary_Metric': 'R²',
                'Mean': f"{stats['r2_mean']:.3f}",
                'Std': f"{stats['r2_std']:.3f}",
                'Min': f"{stats['r2_min']:.3f}",
                'Max': f"{stats['r2_max']:.3f}",
                'N_Models': stats['n_models'],
                'Uncertainty_Level': 'High' if stats['r2_std'] > 0.1 else 'Medium' if stats['r2_std'] > 0.05 else 'Low'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存表格
        summary_file = self.output_dir / 'uncertainty_summary_table.csv'
        summary_df.to_csv(summary_file, index=False)
        
        return summary_df
    
    def run_analysis(self):
        """运行完整分析"""
        logger.info("🚀 Starting A2 uncertainty quantification analysis...")
        
        # 1. 加载数据
        df = self.load_all_metrics()
        logger.info(f"Loaded data from {len(df)} experiments")
        
        # 2. 计算不确定性统计
        uncertainty_stats = self.compute_uncertainty_statistics(df)
        
        # 3. 创建可视化
        self.create_uncertainty_plots(uncertainty_stats)
        
        # 4. 生成摘要表格
        summary_df = self.generate_summary_table(uncertainty_stats)
        
        # 5. 保存完整结果
        results = {
            'raw_data': df.to_dict('records'),
            'uncertainty_statistics': uncertainty_stats
        }
        
        results_file = self.output_dir / 'a2_uncertainty_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 6. 打印摘要
        print("\n" + "="*80)
        print("A2 UNCERTAINTY QUANTIFICATION ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nAnalyzed {len(df)} models with deep ensemble approach")
        
        n_cls = len(uncertainty_stats['classification'])
        n_reg = len(uncertainty_stats['regression'])
        print(f"Classification endpoints: {n_cls}")
        print(f"Regression endpoints: {n_reg}")
        
        if n_cls > 0:
            avg_auc_uncertainty = np.mean([stats['auc_std'] for stats in uncertainty_stats['classification'].values()])
            print(f"Average AUC epistemic uncertainty: {avg_auc_uncertainty:.3f}")
        
        if n_reg > 0:
            avg_r2_uncertainty = np.mean([stats['r2_std'] for stats in uncertainty_stats['regression'].values()])
            print(f"Average R² epistemic uncertainty: {avg_r2_uncertainty:.3f}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print("Generated files:")
        print("  - uncertainty_summary_table.csv")
        print("  - classification_uncertainty.png")
        print("  - regression_uncertainty.png")
        print("  - a2_uncertainty_results.json")
        
        logger.info("✅ A2 analysis completed successfully!")

def main():
    analyzer = SimpleUncertaintyAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
