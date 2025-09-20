#!/usr/bin/env python3
"""
A5: Detailed metrics analysis and class imbalance handling
Comprehensive evaluation with Precision/Recall/F1/MCC/PR-AUC and case studies
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc, average_precision_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetailedMetricsAnalyzer:
    """Comprehensive metrics analysis for toxicity prediction."""
    
    def __init__(self):
        self.endpoint_names = self._load_endpoint_names()
        
    def _load_endpoint_names(self) -> Dict:
        """Load endpoint names."""
        classification_endpoints = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
            "SR-ATAD5-2", "SR-HSE-2", "SR-MMP-2", "SR-p53-2", "NR-ER-2", "NR-AR-2",
            "NR-AhR-2", "NR-Aromatase-2", "NR-PPAR-gamma-2", "SR-ARE-2", "NR-ER-LBD-2",
            "NR-AR-LBD-2", "SR-ATAD5-3", "SR-HSE-3"
        ]
        
        regression_endpoints = [
            "IGC50", "LC50", "LC50DM", "LLNA", "LOAEL"
        ]
        
        return {
            'classification': classification_endpoints,
            'regression': regression_endpoints
        }
    
    def compute_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_prob: np.ndarray = None) -> Dict:
        """Compute comprehensive classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # ROC and PR curves
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            metrics['auc_roc'] = auc(fpr, tpr)
            
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
            metrics['auc_pr'] = auc(recall_curve, precision_curve)
            metrics['average_precision'] = average_precision_score(y_true, y_prob)
            
            # Store curves for plotting
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
            metrics['pr_curve'] = {'precision': precision_curve.tolist(), 'recall': recall_curve.tolist()}
        
        return metrics
    
    def compute_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute comprehensive regression metrics."""
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        from scipy.stats import pearsonr, spearmanr
        
        metrics = {}
        
        # Basic metrics
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p
        
        # Residual analysis
        residuals = y_true - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        
        return metrics
    
    def analyze_class_imbalance(self, y_true: np.ndarray, endpoint_name: str) -> Dict:
        """Analyze class imbalance for a classification endpoint."""
        unique, counts = np.unique(y_true, return_counts=True)
        
        analysis = {
            'endpoint_name': endpoint_name,
            'total_samples': len(y_true),
            'class_distribution': dict(zip(unique.astype(int), counts.astype(int))),
            'imbalance_ratio': None,
            'minority_class_ratio': None
        }
        
        if len(unique) == 2:
            neg_count = counts[unique == 0][0] if 0 in unique else 0
            pos_count = counts[unique == 1][0] if 1 in unique else 0
            
            if pos_count > 0 and neg_count > 0:
                analysis['imbalance_ratio'] = max(neg_count, pos_count) / min(neg_count, pos_count)
                analysis['minority_class_ratio'] = min(neg_count, pos_count) / (neg_count + pos_count)
                analysis['positive_rate'] = pos_count / (neg_count + pos_count)
        
        return analysis
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray, 
                              metric: str = 'f1') -> Tuple[float, Dict]:
        """Find optimal classification threshold."""
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'youden':
                # Youden's J statistic
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        # Compute metrics at optimal threshold
        y_pred_optimal = (y_prob >= best_threshold).astype(int)
        optimal_metrics = self.compute_classification_metrics(y_true, y_pred_optimal, y_prob)
        
        return best_threshold, {
            'threshold': best_threshold,
            'score': best_score,
            'metric_used': metric,
            'metrics_at_threshold': optimal_metrics,
            'threshold_curve': {'thresholds': thresholds.tolist(), 'scores': scores}
        }
    
    def generate_case_studies(self, smiles_list: List[str], y_true: np.ndarray, 
                            y_pred: np.ndarray, y_prob: np.ndarray = None, 
                            n_cases: int = 5) -> Dict:
        """Generate case studies for false positives and false negatives."""
        cases = {
            'true_positives': [],
            'true_negatives': [],
            'false_positives': [],
            'false_negatives': []
        }
        
        for i, (smiles, true_label, pred_label) in enumerate(zip(smiles_list, y_true, y_pred)):
            prob = y_prob[i] if y_prob is not None else None
            
            case_info = {
                'smiles': smiles,
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
                'probability': float(prob) if prob is not None else None,
                'index': i
            }
            
            if true_label == 1 and pred_label == 1:
                cases['true_positives'].append(case_info)
            elif true_label == 0 and pred_label == 0:
                cases['true_negatives'].append(case_info)
            elif true_label == 0 and pred_label == 1:
                cases['false_positives'].append(case_info)
            elif true_label == 1 and pred_label == 0:
                cases['false_negatives'].append(case_info)
        
        # Sort by probability and select top cases
        for case_type in cases:
            if cases[case_type] and y_prob is not None:
                if case_type in ['false_positives', 'true_positives']:
                    # Sort by highest probability
                    cases[case_type] = sorted(cases[case_type], 
                                            key=lambda x: x['probability'], reverse=True)
                else:
                    # Sort by lowest probability
                    cases[case_type] = sorted(cases[case_type], 
                                            key=lambda x: x['probability'])
                
                cases[case_type] = cases[case_type][:n_cases]
        
        return cases
    
    def create_comprehensive_report(self, results_data: Dict) -> Dict:
        """Create comprehensive analysis report."""
        logger.info("📊 Creating comprehensive metrics report...")
        
        report = {
            'analysis_type': 'detailed_metrics_analysis',
            'endpoints': {},
            'summary': {
                'classification': {},
                'regression': {}
            }
        }
        
        # Analyze each endpoint
        for endpoint_type in ['classification', 'regression']:
            endpoint_names = self.endpoint_names[endpoint_type]
            
            for i, endpoint_name in enumerate(endpoint_names):
                if f'{endpoint_type}_endpoint_{i}' in results_data:
                    endpoint_data = results_data[f'{endpoint_type}_endpoint_{i}']
                    
                    endpoint_analysis = {
                        'endpoint_name': endpoint_name,
                        'endpoint_type': endpoint_type,
                        'endpoint_index': i
                    }
                    
                    if endpoint_type == 'classification':
                        # Classification analysis
                        y_true = endpoint_data.get('y_true', [])
                        y_pred = endpoint_data.get('y_pred', [])
                        y_prob = endpoint_data.get('y_prob', [])
                        
                        if y_true and y_pred:
                            # Basic metrics
                            endpoint_analysis['metrics'] = self.compute_classification_metrics(
                                np.array(y_true), np.array(y_pred), 
                                np.array(y_prob) if y_prob else None
                            )
                            
                            # Class imbalance analysis
                            endpoint_analysis['imbalance_analysis'] = self.analyze_class_imbalance(
                                np.array(y_true), endpoint_name
                            )
                            
                            # Optimal threshold
                            if y_prob:
                                threshold_f1, threshold_analysis = self.find_optimal_threshold(
                                    np.array(y_true), np.array(y_prob), 'f1'
                                )
                                endpoint_analysis['optimal_threshold'] = threshold_analysis
                            
                            # Case studies
                            if 'smiles' in endpoint_data:
                                endpoint_analysis['case_studies'] = self.generate_case_studies(
                                    endpoint_data['smiles'], np.array(y_true), 
                                    np.array(y_pred), np.array(y_prob) if y_prob else None
                                )
                    
                    else:
                        # Regression analysis
                        y_true = endpoint_data.get('y_true', [])
                        y_pred = endpoint_data.get('y_pred', [])
                        
                        if y_true and y_pred:
                            endpoint_analysis['metrics'] = self.compute_regression_metrics(
                                np.array(y_true), np.array(y_pred)
                            )
                    
                    report['endpoints'][f'{endpoint_name}'] = endpoint_analysis
        
        # Generate summary statistics
        self._generate_summary_statistics(report)
        
        return report
    
    def _generate_summary_statistics(self, report: Dict):
        """Generate summary statistics across all endpoints."""
        cls_metrics = []
        reg_metrics = []
        
        for endpoint_name, endpoint_data in report['endpoints'].items():
            if endpoint_data['endpoint_type'] == 'classification':
                if 'metrics' in endpoint_data:
                    cls_metrics.append(endpoint_data['metrics'])
            else:
                if 'metrics' in endpoint_data:
                    reg_metrics.append(endpoint_data['metrics'])
        
        # Classification summary
        if cls_metrics:
            cls_summary = {}
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'mcc', 'auc_roc', 'auc_pr']:
                values = [m.get(metric, 0) for m in cls_metrics if metric in m]
                if values:
                    cls_summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'n_endpoints': len(values)
                    }
            report['summary']['classification'] = cls_summary
        
        # Regression summary
        if reg_metrics:
            reg_summary = {}
            for metric in ['r2', 'rmse', 'mae', 'pearson_r', 'spearman_r']:
                values = [m.get(metric, 0) for m in reg_metrics if metric in m]
                if values:
                    reg_summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'n_endpoints': len(values)
                    }
            report['summary']['regression'] = reg_summary

def main():
    """Main function for detailed metrics analysis."""
    logger.info("🚀 A5: Detailed Metrics Analysis and Class Imbalance")
    logger.info("="*60)
    
    # Initialize analyzer
    analyzer = DetailedMetricsAnalyzer()
    
    # This would typically load actual model results
    # For now, create a template structure
    logger.info("📋 Creating detailed metrics analysis template...")
    
    # Template results data (to be populated with actual model outputs)
    template_results = {
        'classification_endpoint_0': {
            'y_true': [0, 1, 0, 1, 1, 0, 1, 0],  # Example data
            'y_pred': [0, 1, 0, 0, 1, 0, 1, 1],
            'y_prob': [0.2, 0.8, 0.3, 0.4, 0.9, 0.1, 0.7, 0.6],
            'smiles': ['CCO', 'CCC', 'CCCO', 'CCCC', 'CCCCO', 'CCCCC', 'CCCCCO', 'CCCCCC']
        }
    }
    
    # Generate comprehensive report
    report = analyzer.create_comprehensive_report(template_results)
    
    # Save report
    output_dir = Path("detailed_metrics_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "detailed_metrics_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"💾 Report saved to: {output_dir}")
    
    # Print summary
    logger.info("\n📊 DETAILED METRICS ANALYSIS SUMMARY")
    logger.info("="*60)
    
    if 'classification' in report['summary']:
        cls_summary = report['summary']['classification']
        logger.info("🎯 Classification Summary:")
        for metric, stats in cls_summary.items():
            logger.info(f"   {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    if 'regression' in report['summary']:
        reg_summary = report['summary']['regression']
        logger.info("🎯 Regression Summary:")
        for metric, stats in reg_summary.items():
            logger.info(f"   {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    logger.info("\n📝 Next steps:")
    logger.info("   1. Populate with actual model results")
    logger.info("   2. Generate visualization plots")
    logger.info("   3. Create case study analysis")
    logger.info("   4. Implement class imbalance handling strategies")

if __name__ == "__main__":
    main()
