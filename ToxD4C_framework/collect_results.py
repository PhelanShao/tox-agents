#!/usr/bin/env python3
"""
Results collection script for ToxD4C experiments
Collects and analyzes results from completed experiments.
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_experiment_results():
    """Collect results from all completed experiments."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        logger.warning("No experiments directory found")
        return []
    
    results = []
    
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_info = {
            'experiment_name': exp_dir.name,
            'start_time': datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat(),
            'status': 'unknown',
            'config': None,
            'final_metrics': None,
            'checkpoints': []
        }
        
        # Check for config file
        config_file = exp_dir / "checkpoints" / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    exp_info['config'] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config for {exp_dir.name}: {e}")
        
        # Check for results file
        results_file = exp_dir / "checkpoints" / f"{exp_dir.name.split('_')[0]}_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    result_data = json.load(f)
                    exp_info['final_metrics'] = result_data.get('final_metrics')
                    exp_info['status'] = 'completed'
            except Exception as e:
                logger.warning(f"Failed to load results for {exp_dir.name}: {e}")
        
        # Check for checkpoints
        checkpoint_dir = exp_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            exp_info['checkpoints'] = [cp.name for cp in checkpoints]
            if checkpoints and exp_info['status'] == 'unknown':
                exp_info['status'] = 'in_progress'
        
        results.append(exp_info)
    
    return results

def analyze_ablation_results(results):
    """Analyze ablation study results."""
    ablation_results = []
    baseline_results = []
    
    for result in results:
        exp_name = result['experiment_name']
        if 'ablation' in exp_name:
            ablation_results.append(result)
        elif 'baseline' in exp_name or 'reproducible' in exp_name:
            baseline_results.append(result)
    
    logger.info(f"Found {len(ablation_results)} ablation experiments")
    logger.info(f"Found {len(baseline_results)} baseline experiments")
    
    # Analyze completed experiments
    completed_ablations = [r for r in ablation_results if r['status'] == 'completed']
    completed_baselines = [r for r in baseline_results if r['status'] == 'completed']
    
    if completed_ablations:
        logger.info("Completed ablation experiments:")
        for exp in completed_ablations:
            logger.info(f"  - {exp['experiment_name']}")
            if exp['final_metrics']:
                # Extract key metrics
                metrics = exp['final_metrics']
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"    {key}: {value:.4f}")
    
    if completed_baselines:
        logger.info("Completed baseline experiments:")
        for exp in completed_baselines:
            logger.info(f"  - {exp['experiment_name']}")
            if exp['final_metrics']:
                metrics = exp['final_metrics']
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"    {key}: {value:.4f}")
    
    return {
        'ablation_experiments': ablation_results,
        'baseline_experiments': baseline_results,
        'completed_ablations': completed_ablations,
        'completed_baselines': completed_baselines
    }

def create_results_summary():
    """Create a comprehensive results summary."""
    logger.info("🔍 Collecting experiment results...")
    
    results = collect_experiment_results()
    
    if not results:
        logger.warning("No experiment results found")
        return
    
    logger.info(f"Found {len(results)} experiments")
    
    # Analyze by status
    status_counts = {}
    for result in results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    logger.info("Experiment status summary:")
    for status, count in status_counts.items():
        logger.info(f"  {status}: {count}")
    
    # Analyze ablation results
    analysis = analyze_ablation_results(results)
    
    # Save summary to file
    summary = {
        'collection_time': datetime.now().isoformat(),
        'total_experiments': len(results),
        'status_counts': status_counts,
        'analysis': analysis,
        'all_results': results
    }
    
    summary_file = Path("experiment_results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"📊 Results summary saved to: {summary_file}")
    
    # Create CSV for easy analysis
    csv_data = []
    for result in results:
        row = {
            'experiment_name': result['experiment_name'],
            'status': result['status'],
            'start_time': result['start_time'],
            'num_checkpoints': len(result['checkpoints'])
        }
        
        # Add config info
        if result['config']:
            config = result['config']
            row.update({
                'batch_size': config.get('batch_size'),
                'hidden_dim': config.get('hidden_dim'),
                'num_encoder_layers': config.get('num_encoder_layers'),
                'use_gnn': config.get('use_gnn'),
                'use_transformer': config.get('use_transformer'),
                'use_geometric_encoding': config.get('use_geometric_encoding'),
                'use_fingerprint_branch': config.get('use_fingerprint_branch')
            })
        
        # Add metrics
        if result['final_metrics']:
            metrics = result['final_metrics']
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        row[f'metric_{key}'] = value
        
        csv_data.append(row)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_file = Path("experiment_results.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"📈 Results CSV saved to: {csv_file}")
    
    return summary

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect ToxD4C experiment results')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results files')
    
    args = parser.parse_args()
    
    # Change to output directory
    if args.output_dir != '.':
        os.chdir(args.output_dir)
    
    summary = create_results_summary()
    
    if summary:
        logger.info("✅ Results collection completed successfully")
        
        # Print quick summary
        print("\n" + "="*60)
        print("📊 EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Collection time: {summary['collection_time']}")
        print("\nStatus breakdown:")
        for status, count in summary['status_counts'].items():
            print(f"  {status}: {count}")
        
        analysis = summary['analysis']
        print(f"\nAblation experiments: {len(analysis['ablation_experiments'])}")
        print(f"Baseline experiments: {len(analysis['baseline_experiments'])}")
        print(f"Completed ablations: {len(analysis['completed_ablations'])}")
        print(f"Completed baselines: {len(analysis['completed_baselines'])}")
        print("="*60)
    else:
        logger.error("❌ Results collection failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
