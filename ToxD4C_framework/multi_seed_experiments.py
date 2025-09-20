#!/usr/bin/env python3
"""
Multi-seed experiments for statistical significance testing
Addresses reviewer requirements R1C5, R1C8, and R2C4
"""

import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiSeedExperimentRunner:
    """Run multiple experiments with different random seeds for statistical analysis."""
    
    def __init__(self, base_config: Dict, seeds: List[int], output_dir: str):
        self.base_config = base_config
        self.seeds = seeds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        
    def run_single_experiment(self, seed: int, experiment_name: str) -> Dict:
        """Run a single experiment with given seed."""
        logger.info(f"🚀 Running experiment with seed {seed}: {experiment_name}")
        
        # Prepare command
        cmd = [
            "python", "train.py",
            "--experiment_name", f"{experiment_name}_seed_{seed}",
            "--seed", str(seed),
            "--deterministic",
            "--use_preprocessed",
            "--preprocessed_dir", "data/data/processed"
        ]
        
        # Add other config parameters
        for key, value in self.base_config.items():
            if key not in ['experiment_name', 'seed']:
                cmd.extend([f"--{key}", str(value)])
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run experiment
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            
            if result.returncode == 0:
                # Parse results from output or result files
                experiment_dir = Path("experiments") / f"{experiment_name}_seed_{seed}"
                results_file = experiment_dir / "final_results.json"
                
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                    
                    elapsed_time = time.time() - start_time
                    results['training_time'] = elapsed_time
                    results['seed'] = seed
                    
                    logger.info(f"✅ Seed {seed} completed in {elapsed_time:.1f}s")
                    return results
                else:
                    logger.warning(f"⚠️  Results file not found for seed {seed}")
                    return None
            else:
                logger.error(f"❌ Seed {seed} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Seed {seed} timed out after 2 hours")
            return None
        except Exception as e:
            logger.error(f"❌ Seed {seed} failed with exception: {e}")
            return None
    
    def run_all_experiments(self, experiment_name: str) -> Dict:
        """Run experiments with all seeds."""
        logger.info(f"🎯 Starting multi-seed experiments: {experiment_name}")
        logger.info(f"   Seeds: {self.seeds}")
        logger.info(f"   Total experiments: {len(self.seeds)}")
        
        all_results = []
        successful_seeds = []
        failed_seeds = []
        
        for seed in self.seeds:
            result = self.run_single_experiment(seed, experiment_name)
            if result is not None:
                all_results.append(result)
                successful_seeds.append(seed)
            else:
                failed_seeds.append(seed)
        
        logger.info(f"📊 Experiment summary:")
        logger.info(f"   Successful: {len(successful_seeds)} / {len(self.seeds)}")
        logger.info(f"   Failed: {len(failed_seeds)}")
        
        if successful_seeds:
            logger.info(f"   Successful seeds: {successful_seeds}")
        if failed_seeds:
            logger.info(f"   Failed seeds: {failed_seeds}")
        
        # Compute statistics
        if len(all_results) >= 2:
            stats = self.compute_statistics(all_results)
            
            # Save results
            output_file = self.output_dir / f"{experiment_name}_multi_seed_results.json"
            final_results = {
                'experiment_name': experiment_name,
                'base_config': self.base_config,
                'seeds': self.seeds,
                'successful_seeds': successful_seeds,
                'failed_seeds': failed_seeds,
                'individual_results': all_results,
                'statistics': stats
            }
            
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info(f"💾 Results saved to: {output_file}")
            return final_results
        else:
            logger.error(f"❌ Insufficient successful experiments ({len(all_results)}) for statistics")
            return None
    
    def compute_statistics(self, results: List[Dict]) -> Dict:
        """Compute statistical metrics across multiple runs."""
        logger.info("📈 Computing statistical metrics...")
        
        stats = {}
        
        # Extract metrics from all runs
        metrics_data = {}
        for result in results:
            if 'test_metrics' in result:
                for metric_name, value in result['test_metrics'].items():
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = []
                    metrics_data[metric_name].append(value)
        
        # Compute statistics for each metric
        for metric_name, values in metrics_data.items():
            if len(values) > 1:
                stats[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'n_runs': len(values),
                    'values': values
                }
                
                # 95% confidence interval
                if len(values) >= 3:
                    from scipy import stats as scipy_stats
                    confidence_interval = scipy_stats.t.interval(
                        0.95, len(values)-1, 
                        loc=np.mean(values), 
                        scale=scipy_stats.sem(values)
                    )
                    stats[metric_name]['ci_95'] = [float(confidence_interval[0]), float(confidence_interval[1])]
        
        return stats

def main():
    """Main function for multi-seed experiments."""
    parser = argparse.ArgumentParser(description='Multi-seed experiments for statistical significance')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 999],
                       help='Random seeds to use')
    parser.add_argument('--experiment_name', type=str, default='toxd4c_multi_seed',
                       help='Base experiment name')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs (reduced for multi-seed)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='multi_seed_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Base configuration
    base_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
    }
    
    logger.info("🎯 Multi-Seed Statistical Experiments")
    logger.info("="*60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Config: {base_config}")
    logger.info(f"Output: {args.output_dir}")
    
    # Run experiments
    runner = MultiSeedExperimentRunner(base_config, args.seeds, args.output_dir)
    results = runner.run_all_experiments(args.experiment_name)
    
    if results:
        logger.info("\n🎉 Multi-seed experiments completed successfully!")
        logger.info("📊 Statistical Summary:")
        
        for metric_name, stats in results['statistics'].items():
            logger.info(f"   {metric_name}:")
            logger.info(f"     Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            if 'ci_95' in stats:
                logger.info(f"     95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
            logger.info(f"     Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    else:
        logger.error("❌ Multi-seed experiments failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
