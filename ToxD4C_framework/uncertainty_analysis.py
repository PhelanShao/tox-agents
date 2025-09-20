#!/usr/bin/env python3
"""
A2 不确定性量化与适用域分析
完成审稿人要求的不确定性评估和适用域定义
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance
from scipy import stats
import lmdb

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置绘图样式
plt.style.use('default')
sns.set_palette("husl")

class UncertaintyAnalyzer:
    """不确定性量化与适用域分析器"""
    
    def __init__(self, experiment_paths: List[str], output_dir: str = "uncertainty_analysis"):
        self.experiment_paths = experiment_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # ToxD4C终点信息
        self.classification_endpoints = [
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
            'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
        ]
        
        self.regression_endpoints = [
            'IGC50', 'LC50', 'LC50DM', 'LLNA', 'LOAEL'
        ]
        
        logger.info(f"Initialized UncertaintyAnalyzer with {len(experiment_paths)} experiments")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_experiment_results(self, exp_path: str) -> Dict:
        """加载单个实验的结果"""
        exp_dir = Path(exp_path)
        
        results = {
            'predictions': {},
            'probabilities': {},
            'targets': {},
            'metrics': {},
            'embeddings': {}
        }
        
        # 查找结果文件
        possible_files = [
            'test_predictions.json',
            'test_results.json', 
            'predictions.json',
            'results.json'
        ]
        
        for filename in possible_files:
            file_path = exp_dir / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    results.update(data)
                break
        
        # 加载指标
        metrics_file = exp_dir / 'test_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                results['metrics'] = json.load(f)
        
        return results
    
    def compute_expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """计算期望校准误差 (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在当前bin中的样本
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def temperature_scaling(self, logits: np.ndarray, labels: np.ndarray, val_logits: np.ndarray) -> Tuple[float, np.ndarray]:
        """温度缩放校准"""
        from scipy.optimize import minimize_scalar
        
        def temperature_scale_loss(temperature):
            scaled_logits = logits / temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))  # sigmoid
            return -np.mean(labels * np.log(scaled_probs + 1e-8) + (1 - labels) * np.log(1 - scaled_probs + 1e-8))
        
        # 在验证集上优化温度
        result = minimize_scalar(temperature_scale_loss, bounds=(0.1, 10.0), method='bounded')
        optimal_temperature = result.x
        
        # 应用温度缩放
        calibrated_logits = val_logits / optimal_temperature
        calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))
        
        return optimal_temperature, calibrated_probs
    
    def deep_ensemble_uncertainty(self, predictions_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """深度集成不确定性"""
        predictions = np.stack(predictions_list, axis=0)  # [n_models, n_samples, n_classes]
        
        # 平均预测
        mean_pred = np.mean(predictions, axis=0)
        
        # 预测不确定性 (epistemic uncertainty)
        epistemic_uncertainty = np.var(predictions, axis=0)
        
        return mean_pred, epistemic_uncertainty
    
    def compute_tanimoto_similarity(self, smiles_list: List[str], train_smiles: List[str]) -> np.ndarray:
        """计算Tanimoto相似性到训练集"""
        logger.info("Computing Tanimoto similarities...")
        
        # 生成指纹
        def get_fingerprint(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        
        # 训练集指纹
        train_fps = []
        for smiles in tqdm(train_smiles, desc="Train fingerprints"):
            fp = get_fingerprint(smiles)
            if fp is not None:
                train_fps.append(fp)
        
        # 计算相似性
        similarities = []
        for smiles in tqdm(smiles_list, desc="Computing similarities"):
            fp = get_fingerprint(smiles)
            if fp is None:
                similarities.append(0.0)
                continue
            
            # 计算到最近k个训练样本的平均相似性
            k = min(5, len(train_fps))
            sims = []
            for train_fp in train_fps:
                sim = DataStructs.TanimotoSimilarity(fp, train_fp)
                sims.append(sim)
            
            # 取top-k平均
            top_k_sims = sorted(sims, reverse=True)[:k]
            avg_sim = np.mean(top_k_sims)
            similarities.append(avg_sim)
        
        return np.array(similarities)
    
    def compute_mahalanobis_distance(self, embeddings: np.ndarray, train_embeddings: np.ndarray) -> np.ndarray:
        """计算马氏距离"""
        logger.info("Computing Mahalanobis distances...")
        
        # 标准化
        scaler = StandardScaler()
        train_embeddings_scaled = scaler.fit_transform(train_embeddings)
        embeddings_scaled = scaler.transform(embeddings)
        
        # 计算协方差矩阵
        cov = EmpiricalCovariance().fit(train_embeddings_scaled)
        
        # 计算马氏距离
        distances = cov.mahalanobis(embeddings_scaled)
        
        return distances
    
    def create_reliability_diagram(self, y_true: np.ndarray, y_prob: np.ndarray, 
                                 endpoint_name: str, save_path: Optional[str] = None):
        """创建可靠性图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 可靠性曲线
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10, normalize=False
        )
        
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title(f"Reliability Diagram - {endpoint_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 预测概率分布
        ax2.hist(y_prob, bins=20, alpha=0.7, density=True, label="Predicted Probabilities")
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Density")
        ax2.set_title(f"Prediction Distribution - {endpoint_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Reliability diagram saved: {save_path}")
        
        plt.show()
        
        return fig
    
    def analyze_classification_uncertainty(self) -> Dict:
        """分析分类任务的不确定性"""
        logger.info("🎯 Analyzing classification uncertainty...")
        
        results = {}
        
        # 加载所有实验结果
        all_experiments = []
        for exp_path in self.experiment_paths:
            exp_results = self.load_experiment_results(exp_path)
            all_experiments.append(exp_results)
        
        # 对每个分类终点进行分析
        for i, endpoint in enumerate(self.classification_endpoints):
            logger.info(f"Analyzing endpoint: {endpoint}")
            
            endpoint_results = {
                'endpoint': endpoint,
                'ece_original': [],
                'ece_calibrated': [],
                'brier_original': [],
                'brier_calibrated': [],
                'temperature': []
            }
            
            # 收集所有模型的预测
            all_predictions = []
            all_probabilities = []
            y_true = None
            
            for exp_results in all_experiments:
                if 'predictions' in exp_results and 'classification' in exp_results['predictions']:
                    pred = exp_results['predictions']['classification'].get(str(i))
                    prob = exp_results['probabilities']['classification'].get(str(i)) if 'probabilities' in exp_results else None
                    
                    if pred is not None:
                        all_predictions.append(np.array(pred))
                        if prob is not None:
                            all_probabilities.append(np.array(prob))
                        
                        # 获取真实标签 (假设所有实验使用相同的测试集)
                        if y_true is None and 'targets' in exp_results:
                            y_true = np.array(exp_results['targets']['classification'].get(str(i), []))
            
            if len(all_predictions) == 0 or y_true is None or len(y_true) == 0:
                logger.warning(f"No valid data for endpoint {endpoint}")
                continue
            
            # 深度集成
            if len(all_probabilities) > 1:
                ensemble_mean, ensemble_uncertainty = self.deep_ensemble_uncertainty(all_probabilities)
                
                # 计算ECE和Brier分数
                ece_original = self.compute_expected_calibration_error(y_true, ensemble_mean)
                brier_original = brier_score_loss(y_true, ensemble_mean)
                
                endpoint_results['ece_original'].append(ece_original)
                endpoint_results['brier_original'].append(brier_original)
                
                # 温度缩放 (使用第一个模型的logits进行演示)
                if len(all_probabilities) > 0:
                    # 这里需要logits，暂时用概率的logit变换
                    logits = np.log(all_probabilities[0] + 1e-8) - np.log(1 - all_probabilities[0] + 1e-8)
                    
                    try:
                        temperature, calibrated_probs = self.temperature_scaling(
                            logits[:len(y_true)//2], y_true[:len(y_true)//2], 
                            logits[len(y_true)//2:]
                        )
                        
                        ece_calibrated = self.compute_expected_calibration_error(
                            y_true[len(y_true)//2:], calibrated_probs
                        )
                        brier_calibrated = brier_score_loss(y_true[len(y_true)//2:], calibrated_probs)
                        
                        endpoint_results['temperature'].append(temperature)
                        endpoint_results['ece_calibrated'].append(ece_calibrated)
                        endpoint_results['brier_calibrated'].append(brier_calibrated)
                        
                    except Exception as e:
                        logger.warning(f"Temperature scaling failed for {endpoint}: {e}")
                
                # 生成可靠性图
                reliability_path = self.output_dir / f"reliability_{endpoint}.png"
                self.create_reliability_diagram(y_true, ensemble_mean, endpoint, str(reliability_path))
            
            results[endpoint] = endpoint_results
        
        return results
    
    def analyze_regression_uncertainty(self) -> Dict:
        """分析回归任务的不确定性"""
        logger.info("📊 Analyzing regression uncertainty...")

        results = {}

        # 加载所有实验结果
        all_experiments = []
        for exp_path in self.experiment_paths:
            exp_results = self.load_experiment_results(exp_path)
            all_experiments.append(exp_results)

        # 对每个回归终点进行分析
        for i, endpoint in enumerate(self.regression_endpoints):
            logger.info(f"Analyzing endpoint: {endpoint}")

            endpoint_results = {
                'endpoint': endpoint,
                'coverage_rates': [],
                'interval_widths': [],
                'prediction_intervals': []
            }

            # 收集所有模型的预测
            all_predictions = []
            y_true = None

            for exp_results in all_experiments:
                if 'predictions' in exp_results and 'regression' in exp_results['predictions']:
                    pred = exp_results['predictions']['regression'].get(str(i))

                    if pred is not None:
                        all_predictions.append(np.array(pred))

                        # 获取真实标签
                        if y_true is None and 'targets' in exp_results:
                            y_true = np.array(exp_results['targets']['regression'].get(str(i), []))

            if len(all_predictions) == 0 or y_true is None or len(y_true) == 0:
                logger.warning(f"No valid data for endpoint {endpoint}")
                continue

            # 深度集成不确定性
            if len(all_predictions) > 1:
                ensemble_mean, ensemble_var = self.deep_ensemble_uncertainty(all_predictions)
                ensemble_std = np.sqrt(ensemble_var)

                # 计算预测区间 (假设正态分布)
                confidence_levels = [0.8, 0.9, 0.95]
                for conf_level in confidence_levels:
                    z_score = stats.norm.ppf((1 + conf_level) / 2)
                    lower_bound = ensemble_mean - z_score * ensemble_std
                    upper_bound = ensemble_mean + z_score * ensemble_std

                    # 计算覆盖率
                    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))

                    # 计算平均区间宽度
                    avg_width = np.mean(upper_bound - lower_bound)

                    endpoint_results['coverage_rates'].append({
                        'confidence_level': conf_level,
                        'coverage_rate': coverage,
                        'interval_width': avg_width
                    })

            results[endpoint] = endpoint_results

        return results

    def analyze_applicability_domain(self, test_smiles: List[str], train_smiles: List[str],
                                   test_embeddings: Optional[np.ndarray] = None,
                                   train_embeddings: Optional[np.ndarray] = None) -> Dict:
        """分析适用域"""
        logger.info("🎯 Analyzing applicability domain...")

        results = {}

        # 1. Tanimoto相似性分析
        if test_smiles and train_smiles:
            tanimoto_similarities = self.compute_tanimoto_similarity(test_smiles, train_smiles)

            # 计算训练集内部相似性分布，确定阈值
            train_internal_sims = self.compute_tanimoto_similarity(train_smiles[:1000], train_smiles)  # 采样避免计算量过大
            threshold_tanimoto = np.percentile(train_internal_sims, 5)  # 5%分位点作为域外阈值

            # 标记域外样本
            out_of_domain_tanimoto = tanimoto_similarities < threshold_tanimoto

            results['tanimoto'] = {
                'similarities': tanimoto_similarities.tolist(),
                'threshold': threshold_tanimoto,
                'out_of_domain_ratio': np.mean(out_of_domain_tanimoto),
                'out_of_domain_indices': np.where(out_of_domain_tanimoto)[0].tolist()
            }

            logger.info(f"Tanimoto analysis: {np.mean(out_of_domain_tanimoto):.1%} samples out of domain")

        # 2. 马氏距离分析
        if test_embeddings is not None and train_embeddings is not None:
            mahalanobis_distances = self.compute_mahalanobis_distance(test_embeddings, train_embeddings)

            # 计算95%置信椭球阈值
            threshold_mahalanobis = np.percentile(mahalanobis_distances, 95)

            # 标记域外样本
            out_of_domain_mahalanobis = mahalanobis_distances > threshold_mahalanobis

            results['mahalanobis'] = {
                'distances': mahalanobis_distances.tolist(),
                'threshold': threshold_mahalanobis,
                'out_of_domain_ratio': np.mean(out_of_domain_mahalanobis),
                'out_of_domain_indices': np.where(out_of_domain_mahalanobis)[0].tolist()
            }

            logger.info(f"Mahalanobis analysis: {np.mean(out_of_domain_mahalanobis):.1%} samples out of domain")

        return results

    def create_umap_visualization(self, embeddings: np.ndarray, labels: np.ndarray,
                                domain_flags: np.ndarray, save_path: Optional[str] = None):
        """创建UMAP嵌入可视化"""
        try:
            import umap
        except ImportError:
            logger.warning("UMAP not installed, skipping visualization")
            return None

        logger.info("Creating UMAP visualization...")

        # UMAP降维
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制域内样本
        in_domain = ~domain_flags
        scatter1 = ax.scatter(embedding_2d[in_domain, 0], embedding_2d[in_domain, 1],
                             c=labels[in_domain], cmap='viridis', alpha=0.6, s=20,
                             label='In Domain')

        # 绘制域外样本
        scatter2 = ax.scatter(embedding_2d[domain_flags, 0], embedding_2d[domain_flags, 1],
                             c=labels[domain_flags], cmap='viridis', alpha=0.8, s=40,
                             marker='^', edgecolors='red', linewidth=1,
                             label='Out of Domain')

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Applicability Domain Analysis - UMAP Embedding')
        ax.legend()

        # 添加颜色条
        plt.colorbar(scatter1, ax=ax, label='Target Value')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"UMAP visualization saved: {save_path}")

        plt.show()

        return fig

def main():
    parser = argparse.ArgumentParser(description='A2 Uncertainty Quantification Analysis')
    parser.add_argument('--experiment_paths', nargs='+', required=True,
                       help='Paths to experiment directories')
    parser.add_argument('--output_dir', type=str, default='uncertainty_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = UncertaintyAnalyzer(args.experiment_paths, args.output_dir)
    
    # 执行分析
    logger.info("🚀 Starting A2 uncertainty quantification analysis...")
    
    # 分类不确定性分析
    classification_results = analyzer.analyze_classification_uncertainty()
    
    # 回归不确定性分析  
    regression_results = analyzer.analyze_regression_uncertainty()
    
    # 保存结果
    results = {
        'classification': classification_results,
        'regression': regression_results
    }
    
    output_file = analyzer.output_dir / 'uncertainty_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"✅ Analysis complete! Results saved to: {output_file}")
    
    # 生成摘要报告
    logger.info("\n" + "="*60)
    logger.info("UNCERTAINTY ANALYSIS SUMMARY")
    logger.info("="*60)
    
    for endpoint, result in classification_results.items():
        if result['ece_original']:
            logger.info(f"{endpoint}:")
            logger.info(f"  ECE (original): {np.mean(result['ece_original']):.3f}")
            logger.info(f"  Brier (original): {np.mean(result['brier_original']):.3f}")
            if result['ece_calibrated']:
                logger.info(f"  ECE (calibrated): {np.mean(result['ece_calibrated']):.3f}")

if __name__ == "__main__":
    main()
