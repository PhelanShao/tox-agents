#!/usr/bin/env python3
"""
Single endpoint vs aggregated scores training for R1C3 sensitivity analysis
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_single_endpoint_configs():
    """Create configurations for single endpoint training."""
    
    # Based on the analysis, we have 26 classification and 5 regression endpoints
    configs = {
        'aggregated_all': {
            'description': 'All endpoints (baseline)',
            'endpoint_selection': 'all',
            'disable_flags': []
        },
        'classification_only': {
            'description': 'Classification endpoints only',
            'endpoint_selection': 'classification',
            'disable_flags': ['--disable_regression']
        },
        'regression_only': {
            'description': 'Regression endpoints only', 
            'endpoint_selection': 'regression',
            'disable_flags': ['--disable_classification']
        }
    }
    
    # Add high-coverage classification endpoints (>20% data coverage)
    high_coverage_cls_endpoints = [1, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    
    for endpoint_idx in high_coverage_cls_endpoints:
        configs[f'single_cls_{endpoint_idx}'] = {
            'description': f'Single classification endpoint {endpoint_idx}',
            'endpoint_selection': f'single_cls_{endpoint_idx}',
            'disable_flags': ['--disable_regression', f'--single_endpoint_cls={endpoint_idx}']
        }
    
    # Add regression endpoints
    for endpoint_idx in range(5):
        configs[f'single_reg_{endpoint_idx}'] = {
            'description': f'Single regression endpoint {endpoint_idx}',
            'endpoint_selection': f'single_reg_{endpoint_idx}',
            'disable_flags': ['--disable_classification', f'--single_endpoint_reg={endpoint_idx}']
        }
    
    return configs

def run_sensitivity_analysis():
    """Run R1C3 sensitivity analysis comparing single endpoints vs aggregated scores."""
    
    logger.info("🔬 R1C3 Sensitivity Analysis: Single Endpoints vs Aggregated Scores")
    logger.info("="*70)
    
    configs = create_single_endpoint_configs()
    seeds = [42, 123, 456]  # Reduced for faster analysis
    
    base_config = {
        'batch_size': 16,
        'num_epochs': 10,  # Reduced for sensitivity analysis
        'learning_rate': 1e-4,
        'use_preprocessed': True,
        'preprocessed_dir': 'data/data/processed',
        'deterministic': True
    }
    
    logger.info(f"Configurations: {len(configs)}")
    logger.info(f"Seeds per config: {len(seeds)}")
    logger.info(f"Total experiments: {len(configs) * len(seeds)}")
    
    all_results = {}
    
    # Run key comparisons first
    priority_configs = ['aggregated_all', 'classification_only', 'regression_only', 
                       'single_cls_1', 'single_cls_3', 'single_reg_0', 'single_reg_1']
    
    for config_name in priority_configs:
        if config_name not in configs:
            continue
            
        config_info = configs[config_name]
        logger.info(f"\n🧪 Running: {config_name}")
        logger.info(f"   Description: {config_info['description']}")
        
        config_results = []
        
        for seed in seeds:
            logger.info(f"   🎲 Seed {seed}...")
            
            # Prepare command
            cmd = [
                "python", "train.py",
                "--experiment_name", f"r1c3_{config_name}_seed_{seed}",
                "--seed", str(seed)
            ]
            
            # Add base config
            for key, value in base_config.items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key}")
                else:
                    cmd.extend([f"--{key}", str(value)])
            
            # Add specific flags
            cmd.extend(config_info['disable_flags'])
            
            logger.info(f"     Command: {' '.join(cmd[-10:])}")  # Show last 10 args
            
            # Run experiment
            start_time = time.time()
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
                
                if result.returncode == 0:
                    # Try to load results
                    experiment_dir = Path("experiments") / f"r1c3_{config_name}_seed_{seed}"
                    results_file = experiment_dir / "final_results.json"
                    
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            exp_results = json.load(f)
                        
                        exp_results['seed'] = seed
                        exp_results['training_time'] = time.time() - start_time
                        exp_results['config_name'] = config_name
                        
                        config_results.append(exp_results)
                        logger.info(f"     ✅ Completed in {exp_results['training_time']:.1f}s")
                    else:
                        logger.warning(f"     ⚠️  Results file not found")
                else:
                    logger.error(f"     ❌ Failed with return code {result.returncode}")
                    if result.stderr:
                        logger.error(f"     Error: {result.stderr[:200]}...")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"     ❌ Timed out after 30 minutes")
            except Exception as e:
                logger.error(f"     ❌ Exception: {e}")
        
        all_results[config_name] = {
            'description': config_info['description'],
            'endpoint_selection': config_info['endpoint_selection'],
            'results': config_results,
            'n_successful': len(config_results),
            'n_total': len(seeds)
        }
        
        logger.info(f"   📊 {config_name}: {len(config_results)}/{len(seeds)} successful")
    
    # Compute comparative statistics
    logger.info("\n📈 Computing sensitivity analysis statistics...")
    
    sensitivity_stats = {}
    
    for config_name, config_data in all_results.items():
        if config_data['n_successful'] >= 1:
            results = config_data['results']
            
            # Aggregate metrics across seeds
            metrics_data = {}
            for result in results:
                if 'test_metrics' in result:
                    for metric_name, value in result['test_metrics'].items():
                        if metric_name not in metrics_data:
                            metrics_data[metric_name] = []
                        metrics_data[metric_name].append(value)
            
            # Compute statistics
            config_stats = {}
            for metric_name, values in metrics_data.items():
                config_stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)) if len(values) > 1 else 0.0,
                    'n_runs': len(values),
                    'values': values
                }
            
            sensitivity_stats[config_name] = config_stats
    
    # Perform sensitivity comparisons
    logger.info("🔬 Performing sensitivity comparisons...")
    
    comparisons = {
        'aggregated_vs_classification_only': {
            'baseline': 'aggregated_all',
            'comparison': 'classification_only',
            'description': 'All endpoints vs Classification only'
        },
        'aggregated_vs_regression_only': {
            'baseline': 'aggregated_all', 
            'comparison': 'regression_only',
            'description': 'All endpoints vs Regression only'
        },
        'aggregated_vs_single_cls': {
            'baseline': 'aggregated_all',
            'comparison': 'single_cls_1',
            'description': 'All endpoints vs Single classification endpoint'
        },
        'aggregated_vs_single_reg': {
            'baseline': 'aggregated_all',
            'comparison': 'single_reg_0', 
            'description': 'All endpoints vs Single regression endpoint'
        }
    }
    
    sensitivity_comparisons = {}
    
    for comp_name, comp_info in comparisons.items():
        baseline = comp_info['baseline']
        comparison = comp_info['comparison']
        
        if baseline in sensitivity_stats and comparison in sensitivity_stats:
            baseline_stats = sensitivity_stats[baseline]
            comparison_stats = sensitivity_stats[comparison]
            
            sensitivity_comparisons[comp_name] = {
                'description': comp_info['description'],
                'baseline': baseline,
                'comparison': comparison,
                'metrics': {}
            }
            
            for metric_name in baseline_stats.keys():
                if metric_name in comparison_stats:
                    baseline_mean = baseline_stats[metric_name]['mean']
                    comparison_mean = comparison_stats[metric_name]['mean']
                    
                    sensitivity_comparisons[comp_name]['metrics'][metric_name] = {
                        'baseline_mean': baseline_mean,
                        'comparison_mean': comparison_mean,
                        'absolute_difference': comparison_mean - baseline_mean,
                        'relative_difference': (comparison_mean - baseline_mean) / baseline_mean if baseline_mean != 0 else 0,
                        'performance_change': 'improvement' if comparison_mean > baseline_mean else 'degradation'
                    }
    
    # Save comprehensive results
    output_dir = Path("r1c3_sensitivity_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_results = {
        'experiment_type': 'r1c3_sensitivity_analysis',
        'description': 'Single endpoints vs aggregated scores comparison',
        'base_config': base_config,
        'seeds': seeds,
        'configurations': configs,
        'individual_results': all_results,
        'sensitivity_statistics': sensitivity_stats,
        'sensitivity_comparisons': sensitivity_comparisons,
        'summary': {
            'total_experiments': len(priority_configs) * len(seeds),
            'successful_configs': len(sensitivity_stats),
            'key_findings': {}
        }
    }
    
    # Generate key findings
    if sensitivity_comparisons:
        key_findings = {}
        for comp_name, comp_data in sensitivity_comparisons.items():
            if 'classification_auc' in comp_data['metrics']:
                auc_change = comp_data['metrics']['classification_auc']['relative_difference']
                key_findings[comp_name] = {
                    'auc_relative_change': auc_change,
                    'interpretation': 'significant' if abs(auc_change) > 0.05 else 'minimal'
                }
        
        final_results['summary']['key_findings'] = key_findings
    
    output_file = output_dir / "r1c3_sensitivity_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info(f"💾 Results saved to: {output_file}")
    
    # Print summary
    logger.info("\n📊 R1C3 SENSITIVITY ANALYSIS SUMMARY")
    logger.info("="*60)
    
    for comp_name, comp_data in sensitivity_comparisons.items():
        logger.info(f"\n🔍 {comp_data['description']}:")
        
        if 'classification_auc' in comp_data['metrics']:
            auc_metrics = comp_data['metrics']['classification_auc']
            logger.info(f"   AUC: {auc_metrics['baseline_mean']:.4f} → {auc_metrics['comparison_mean']:.4f}")
            logger.info(f"   Change: {auc_metrics['relative_difference']:.2%} ({auc_metrics['performance_change']})")
        
        if 'regression_r2' in comp_data['metrics']:
            r2_metrics = comp_data['metrics']['regression_r2']
            logger.info(f"   R²: {r2_metrics['baseline_mean']:.4f} → {r2_metrics['comparison_mean']:.4f}")
            logger.info(f"   Change: {r2_metrics['relative_difference']:.2%} ({r2_metrics['performance_change']})")
    
    logger.info("\n🎉 R1C3 sensitivity analysis completed!")
    return 0

if __name__ == "__main__":
    exit(run_sensitivity_analysis())
