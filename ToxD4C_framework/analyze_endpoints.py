#!/usr/bin/env python3
"""
Analyze endpoints in ToxD4C dataset for R1C3 and R1C11 requirements
"""

import lmdb
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset_endpoints(lmdb_path: str):
    """Analyze endpoints in the dataset."""
    logger.info(f"🔍 Analyzing endpoints in: {lmdb_path}")
    
    # Open LMDB
    subdir_flag = Path(lmdb_path).is_dir()
    env = lmdb.open(lmdb_path, subdir=subdir_flag, readonly=True, lock=False, readahead=False, meminit=False)
    
    classification_data = []
    regression_data = []
    endpoint_stats = defaultdict(list)
    
    sample_count = 0
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                key_str = key.decode('ascii')
                if key_str.isdigit() or key_str in ['length', '__keys__']:
                    continue
                
                data = pickle.loads(value)
                sample_count += 1
                
                # Extract classification targets
                if 'classification_target' in data:
                    cls_targets = np.array(data['classification_target'])
                    classification_data.append(cls_targets)
                
                # Extract regression targets  
                if 'regression_target' in data:
                    reg_targets = np.array(data['regression_target'])
                    regression_data.append(reg_targets)
                
                if sample_count >= 1000:  # Sample first 1000 for analysis
                    break
                    
            except Exception as e:
                continue
    
    env.close()
    
    # Analyze classification endpoints
    if classification_data:
        cls_array = np.array(classification_data)
        logger.info(f"📊 Classification data shape: {cls_array.shape}")
        
        cls_stats = {}
        for i in range(cls_array.shape[1]):
            endpoint_values = cls_array[:, i]
            valid_mask = endpoint_values != -10000  # Assuming -10000 is missing value
            valid_values = endpoint_values[valid_mask]
            
            if len(valid_values) > 0:
                cls_stats[f'cls_endpoint_{i}'] = {
                    'n_samples': int(len(valid_values)),
                    'n_positive': int(np.sum(valid_values == 1)),
                    'n_negative': int(np.sum(valid_values == 0)),
                    'positive_rate': float(np.mean(valid_values)),
                    'missing_rate': float(1 - len(valid_values) / len(endpoint_values))
                }
    
    # Analyze regression endpoints
    if regression_data:
        reg_array = np.array(regression_data)
        logger.info(f"📊 Regression data shape: {reg_array.shape}")
        
        reg_stats = {}
        for i in range(reg_array.shape[1]):
            endpoint_values = reg_array[:, i]
            valid_mask = endpoint_values != -10000.0  # Assuming -10000.0 is missing value
            valid_values = endpoint_values[valid_mask]
            
            if len(valid_values) > 0:
                reg_stats[f'reg_endpoint_{i}'] = {
                    'n_samples': int(len(valid_values)),
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'missing_rate': float(1 - len(valid_values) / len(endpoint_values))
                }
    
    return {
        'total_samples': sample_count,
        'classification_endpoints': cls_stats if 'cls_stats' in locals() else {},
        'regression_endpoints': reg_stats if 'reg_stats' in locals() else {}
    }

def create_endpoint_analysis_report():
    """Create comprehensive endpoint analysis report."""
    logger.info("📋 Creating endpoint analysis report...")
    
    datasets = ['train', 'valid', 'test']
    all_results = {}
    
    for dataset in datasets:
        lmdb_path = f"data/data/processed/{dataset}.lmdb"
        if Path(lmdb_path).exists():
            logger.info(f"Analyzing {dataset} dataset...")
            results = analyze_dataset_endpoints(lmdb_path)
            all_results[dataset] = results
        else:
            logger.warning(f"Dataset not found: {lmdb_path}")
    
    # Generate summary report
    report = {
        'analysis_type': 'endpoint_analysis',
        'datasets': all_results,
        'summary': {}
    }
    
    # Calculate overall statistics
    if 'train' in all_results:
        train_data = all_results['train']
        
        # Classification summary
        if train_data['classification_endpoints']:
            n_cls_endpoints = len(train_data['classification_endpoints'])
            avg_positive_rate = np.mean([
                stats['positive_rate'] 
                for stats in train_data['classification_endpoints'].values()
            ])
            avg_missing_rate_cls = np.mean([
                stats['missing_rate']
                for stats in train_data['classification_endpoints'].values()
            ])
            
            report['summary']['classification'] = {
                'n_endpoints': n_cls_endpoints,
                'avg_positive_rate': float(avg_positive_rate),
                'avg_missing_rate': float(avg_missing_rate_cls)
            }
        
        # Regression summary
        if train_data['regression_endpoints']:
            n_reg_endpoints = len(train_data['regression_endpoints'])
            avg_missing_rate_reg = np.mean([
                stats['missing_rate']
                for stats in train_data['regression_endpoints'].values()
            ])
            
            report['summary']['regression'] = {
                'n_endpoints': n_reg_endpoints,
                'avg_missing_rate': float(avg_missing_rate_reg)
            }
    
    # Save report
    output_file = "endpoint_analysis_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"💾 Report saved to: {output_file}")
    
    # Print summary
    logger.info("\n📊 ENDPOINT ANALYSIS SUMMARY")
    logger.info("="*50)
    
    if 'classification' in report['summary']:
        cls_summary = report['summary']['classification']
        logger.info(f"Classification endpoints: {cls_summary['n_endpoints']}")
        logger.info(f"Average positive rate: {cls_summary['avg_positive_rate']:.3f}")
        logger.info(f"Average missing rate: {cls_summary['avg_missing_rate']:.3f}")
    
    if 'regression' in report['summary']:
        reg_summary = report['summary']['regression']
        logger.info(f"Regression endpoints: {reg_summary['n_endpoints']}")
        logger.info(f"Average missing rate: {reg_summary['avg_missing_rate']:.3f}")
    
    return report

def create_per_endpoint_metrics_table():
    """Create detailed per-endpoint metrics table for SI."""
    logger.info("📋 Creating per-endpoint metrics table...")
    
    # This would typically load results from trained models
    # For now, create a template structure
    
    # Define endpoint names (you may need to update these based on actual data)
    classification_endpoints = [
        f"NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        f"NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]
    
    regression_endpoints = [
        "IGC50", "LC50", "LC50DM", "LLNA", "LOAEL"
    ]
    
    # Template for per-endpoint results
    per_endpoint_results = {
        'classification_endpoints': {},
        'regression_endpoints': {},
        'metadata': {
            'model_version': 'ToxD4C_v1.0',
            'evaluation_date': '2025-09-03',
            'dataset_split': '8:1:1',
            'n_seeds': 5,
            'statistical_test': 't-test'
        }
    }
    
    # Template metrics for each classification endpoint
    for endpoint in classification_endpoints:
        per_endpoint_results['classification_endpoints'][endpoint] = {
            'metrics': {
                'auc_roc': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'auc_pr': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'accuracy': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'precision': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'recall': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'f1_score': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'specificity': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5}
            },
            'dataset_stats': {
                'n_samples': 0,
                'n_positive': 0,
                'n_negative': 0,
                'positive_rate': 0.0,
                'missing_rate': 0.0
            },
            'statistical_significance': {
                'vs_baseline': {'p_value': 0.0, 'significant': False},
                'vs_random': {'p_value': 0.0, 'significant': False}
            }
        }
    
    # Template metrics for each regression endpoint
    for endpoint in regression_endpoints:
        per_endpoint_results['regression_endpoints'][endpoint] = {
            'metrics': {
                'r2': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'rmse': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'mae': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'pearson_r': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5},
                'spearman_r': {'mean': 0.0, 'std': 0.0, 'ci_95': [0.0, 0.0], 'n_runs': 5}
            },
            'dataset_stats': {
                'n_samples': 0,
                'mean_value': 0.0,
                'std_value': 0.0,
                'min_value': 0.0,
                'max_value': 0.0,
                'missing_rate': 0.0
            },
            'statistical_significance': {
                'vs_baseline': {'p_value': 0.0, 'significant': False},
                'vs_random': {'p_value': 0.0, 'significant': False}
            }
        }
    
    # Save template
    output_file = "per_endpoint_metrics_template.json"
    with open(output_file, 'w') as f:
        json.dump(per_endpoint_results, f, indent=2)
    
    logger.info(f"💾 Per-endpoint metrics template saved to: {output_file}")
    logger.info("📝 This template should be populated with actual training results")
    
    return per_endpoint_results

def main():
    """Main analysis function."""
    logger.info("🚀 ToxD4C Endpoint Analysis for R1C3 and R1C11")
    logger.info("="*60)
    
    # 1. Analyze dataset endpoints
    endpoint_report = create_endpoint_analysis_report()
    
    # 2. Create per-endpoint metrics template
    per_endpoint_template = create_per_endpoint_metrics_table()
    
    logger.info("\n✅ Analysis completed!")
    logger.info("📋 Generated files:")
    logger.info("   - endpoint_analysis_report.json")
    logger.info("   - per_endpoint_metrics_template.json")
    
    logger.info("\n📝 Next steps for R1C3:")
    logger.info("   1. Train models on individual endpoints")
    logger.info("   2. Train models on aggregated scores")
    logger.info("   3. Compare performance (sensitivity analysis)")
    
    logger.info("\n📝 Next steps for R1C11:")
    logger.info("   1. Run multi-seed experiments")
    logger.info("   2. Populate per-endpoint metrics template")
    logger.info("   3. Generate SI tables and figures")

if __name__ == "__main__":
    main()
