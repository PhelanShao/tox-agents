#!/usr/bin/env python3
"""
R1C3 Sensitivity Analysis: Single endpoints vs aggregated scores training
Enhanced version with statistical analysis and visualization
"""

import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from configs.toxd4c_config import CLASSIFICATION_TASKS, REGRESSION_TASKS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    """Comprehensive sensitivity analysis for R1C3 reviewer response."""
    
    def __init__(self, base_config: Dict, output_dir: str = "r1c3_sensitivity_results"):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.comparison_stats = {}
        # Always define to avoid attribute errors in downstream methods
        self.statistical_tests = {}

    def ingest_baseline_results(self, results_json_path: str, config_name: str = 'multi_task_all', seed: int = 42):
        """Ingest an existing multi-task results.json as baseline without retraining.

        The JSON is expected to be produced by train.py and contain a 'final_metrics' dict
        with keys like 'task_<idx>_auc' and 'task_<idx>_r2'.
        """
        try:
            with open(results_json_path, 'r') as f:
                exp_results = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read baseline results from {results_json_path}: {e}")
            return

        final_metrics = exp_results.get('final_metrics', exp_results)
        run_record = {
            'success': True,
            'config_name': config_name,
            'seed': seed,
            'training_time': 0.0,
            'results': exp_results,
            'stdout': '',
            'final_metrics': final_metrics
        }

        if config_name not in self.results:
            self.results[config_name] = {
                'description': 'Multi-task learning (all endpoints) [ingested]',
                'task_mode': 'multi',
                'successful_runs': [],
                'failed_runs': []
            }
        self.results[config_name]['successful_runs'].append(run_record)
        logger.info(f"📥 Ingested baseline results from: {results_json_path} → as {config_name}")

    def export_single_vs_multi_delta(self, out_csv: Path, baseline_key: str = 'multi_task_all'):
        """Export per-endpoint delta between single-task and multi-task baseline.

        For classification endpoints: delta = AUC_multi[idx] - AUC_single
        For regression endpoints:     delta = R2_multi[idx]  - R2_single
        """
        if baseline_key not in self.results or not self.results[baseline_key]['successful_runs']:
            logger.warning(f"Baseline '{baseline_key}' not found; skip delta export.")
            return

        baseline_metrics = self.results[baseline_key]['successful_runs'][0].get('final_metrics', {})
        rows = []

        # Walk through all recorded single-task runs
        for cfg_name, data in self.results.items():
            if not cfg_name.startswith('single_'):
                continue
            if not data['successful_runs']:
                continue
            fm = data['successful_runs'][0].get('final_metrics', {})

            # Classification single-task
            if cfg_name.startswith('single_cls_'):
                try:
                    idx = int(cfg_name.split('_')[-1])
                except Exception:
                    continue
                ep_name = CLASSIFICATION_TASKS[idx] if idx < len(CLASSIFICATION_TASKS) else f'cls_{idx}'
                single_auc = fm.get('task_0_auc', None)
                multi_auc = baseline_metrics.get(f'task_{idx}_auc', None)
                delta = (multi_auc - single_auc) if (single_auc is not None and multi_auc is not None) else None
                rows.append({
                    'endpoint': ep_name,
                    'type': 'classification',
                    'single': single_auc,
                    'multi_full': multi_auc,
                    'delta': delta
                })

            # Regression single-task
            if cfg_name.startswith('single_reg_'):
                try:
                    idx = int(cfg_name.split('_')[-1])
                except Exception:
                    continue
                ep_name = REGRESSION_TASKS[idx] if idx < len(REGRESSION_TASKS) else f'reg_{idx}'
                single_r2 = fm.get('task_0_r2', None)
                multi_r2 = baseline_metrics.get(f'task_{idx}_r2', None)
                delta = (multi_r2 - single_r2) if (single_r2 is not None and multi_r2 is not None) else None
                rows.append({
                    'endpoint': ep_name,
                    'type': 'regression',
                    'single': single_r2,
                    'multi_full': multi_r2,
                    'delta': delta
                })

        if not rows:
            logger.warning("No single-task runs found for delta export.")
            return

        df = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.info(f"📤 Single vs Multi delta exported to: {out_csv}")
        
    def create_experiment_configs(
        self,
        skip_multitask: bool = False,
        cover_all_cls: bool = False,
        cover_all_reg: bool = False,
        head_only_ft: bool = False,
        resume_from_ckpt: str = None,
        head_ft_epochs: int = 3,
        head_ft_lr: float = 5e-4,
        head_ft_warmup: float = 0.02,
    ) -> Dict[str, Dict]:
        """Create comprehensive experiment configurations."""

        configs = {}
        
        # Skip multi-task baseline if already exists (to save time)
        if not skip_multitask:
            configs.update({
                'multi_task_all': {
                    'description': 'Multi-task learning (all endpoints)',
                    'task_mode': 'multi',
                    'args': []
                },
                'aggregated_all': {
                    'description': 'Aggregated scoring (all endpoints)',
                    'task_mode': 'aggregated', 
                    'args': []
                },
                
                # Task type separation
                'classification_only': {
                    'description': 'Classification endpoints only',
                    'task_mode': 'multi',
                    'args': ['--disable_regression']
                },
                'regression_only': {
                    'description': 'Regression endpoints only',
                    'task_mode': 'multi',
                    'args': ['--disable_classification']
                }
            })
        else:
            logger.info("⏭️ Skipping multi-task baselines (assuming existing results available)")
        
        # Add single classification endpoints
        cls_indices = list(range(len(CLASSIFICATION_TASKS))) if cover_all_cls else [0, 1, 3, 7, 8, 9, 10]
        for idx in cls_indices:
            if idx < len(CLASSIFICATION_TASKS):
                args = ['--task_mode', 'single', '--single_endpoint_cls', str(idx)]
                if head_only_ft and resume_from_ckpt:
                    args += ['--resume_from', resume_from_ckpt, '--freeze_trunk',
                             '--num_epochs', str(head_ft_epochs),
                             '--learning_rate', str(head_ft_lr),
                             '--warmup_ratio', str(head_ft_warmup)]
                configs[f'single_cls_{idx}'] = {
                    'description': f'Single classification: {CLASSIFICATION_TASKS[idx]}',
                    'task_mode': 'single',
                    'args': args
                }

        # Add regression endpoints (only 5 total)
        reg_indices = list(range(len(REGRESSION_TASKS))) if cover_all_reg else list(range(len(REGRESSION_TASKS)))
        for idx in reg_indices:
            args = ['--task_mode', 'single', '--single_endpoint_reg', str(idx)]
            if head_only_ft and resume_from_ckpt:
                args += ['--resume_from', resume_from_ckpt, '--freeze_trunk',
                         '--num_epochs', str(head_ft_epochs),
                         '--learning_rate', str(head_ft_lr),
                         '--warmup_ratio', str(head_ft_warmup)]
            configs[f'single_reg_{idx}'] = {
                'description': f'Single regression: {REGRESSION_TASKS[idx]}',
                'task_mode': 'single',
                'args': args
            }
            
        return configs
    
    def run_single_experiment(self, config_name: str, config_info: Dict, seed: int) -> Dict:
        """Run a single training experiment."""
        
        experiment_name = f"r1c3_{config_name}_seed_{seed}"
        logger.info(f"🧪 Running {experiment_name}")
        
        # Prepare command
        cmd = [
            "python", "train.py",
            "--experiment_name", experiment_name,
            "--seed", str(seed)
        ]
        
        # Add base configuration
        for key, value in self.base_config.items():
            if isinstance(value, bool) and value:
                cmd.append(f"--{key}")
            elif not isinstance(value, bool):
                cmd.extend([f"--{key}", str(value)])
        
        # Add experiment-specific arguments
        cmd.extend(config_info['args'])
        
        # Reuse cached results if available to avoid re-training
        try:
            base_dir = Path(__file__).parent
            exp_dirs = sorted(
                (base_dir / "experiments").glob(f"{experiment_name}_*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if exp_dirs:
                checkpoints = exp_dirs[0] / "checkpoints"
                results_file = checkpoints / f"{experiment_name}_results.json"
                if results_file.exists():
                    with open(results_file, 'r') as f:
                        exp_results = json.load(f)
                    logger.info(f"♻️ Using cached results for {experiment_name}")
                    return {
                        'success': True,
                        'config_name': config_name,
                        'seed': seed,
                        'training_time': 0.0,
                        'results': exp_results,
                        'stdout': "",
                        'final_metrics': exp_results.get('final_metrics', {})
                    }
        except Exception:
            # Non-fatal; fall back to training
            pass

        start_time = time.time()
        
        try:
            # Run with timeout
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                # No timeout - let training complete naturally
                cwd=Path(__file__).parent
            )
            
            if result.returncode == 0:
                # Parse results
                results_pattern = Path("experiments") / f"{experiment_name}_*" / "checkpoints"
                experiment_dirs = list(results_pattern.parent.parent.glob(f"{experiment_name}_*"))
                
                if experiment_dirs:
                    results_file = experiment_dirs[0] / "checkpoints" / f"{experiment_name}_results.json"
                    
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            exp_results = json.load(f)
                        
                        return {
                            'success': True,
                            'config_name': config_name,
                            'seed': seed,
                            'training_time': time.time() - start_time,
                            'results': exp_results,
                            'stdout': result.stdout[-1000:] if result.stdout else "",
                            'final_metrics': exp_results.get('final_metrics', {})
                        }
            
            return {
                'success': False,
                'config_name': config_name,
                'seed': seed,
                'error': f"Return code: {result.returncode}",
                'stderr': result.stderr[-500:] if result.stderr else ""
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'config_name': config_name,
                'seed': seed,
                'error': "Unexpected timeout"
            }
        except Exception as e:
            return {
                'success': False,
                'config_name': config_name,
                'seed': seed,
                'error': str(e)
            }
    
    def run_sensitivity_analysis(self,
        seeds: List[int] = [42, 123, 456],
        max_workers: int = 2,
        skip_multitask: bool = False,
        parallel_single_endpoints: bool = False,
        postprocess_only: bool = False,
        cover_all_cls: bool = False,
        cover_all_reg: bool = False,
        head_only_ft: bool = False,
        resume_from_ckpt: str = None,
        head_ft_epochs: int = 3,
        head_ft_lr: float = 5e-4,
        head_ft_warmup: float = 0.02,
    ):
        """Run comprehensive sensitivity analysis."""
        
        logger.info("🔬 Starting R1C3 Sensitivity Analysis")
        logger.info(f"Seeds: {seeds}")
        logger.info(f"Max parallel workers: {max_workers}")
        
        configs = self.create_experiment_configs(
            skip_multitask=skip_multitask,
            cover_all_cls=cover_all_cls,
            cover_all_reg=cover_all_reg,
            head_only_ft=head_only_ft,
            resume_from_ckpt=resume_from_ckpt,
            head_ft_epochs=head_ft_epochs,
            head_ft_lr=head_ft_lr,
            head_ft_warmup=head_ft_warmup,
        )
        total_experiments = len(configs) * len(seeds)
        
        logger.info(f"Total experiments: {total_experiments}")
        logger.info("="*60)
        
        # Prepare experiments
        experiments = []
        for config_name, config_info in configs.items():
            for seed in seeds:
                experiments.append((config_name, config_info, seed))
        
        # Run experiments or load cached results
        completed_experiments = []
        
        if postprocess_only:
            base_dir = Path(__file__).parent
            for config_name, _config_info in configs.items():
                for seed in seeds:
                    experiment_name = f"r1c3_{config_name}_seed_{seed}"
                    exp_dirs = sorted(
                        (base_dir / "experiments").glob(f"{experiment_name}_*"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True
                    )
                    if exp_dirs:
                        checkpoints = exp_dirs[0] / "checkpoints"
                        results_file = checkpoints / f"{experiment_name}_results.json"
                        if results_file.exists():
                            try:
                                with open(results_file, 'r') as f:
                                    exp_results = json.load(f)
                                completed_experiments.append({
                                    'success': True,
                                    'config_name': config_name,
                                    'seed': seed,
                                    'training_time': 0.0,
                                    'results': exp_results,
                                    'stdout': "",
                                    'final_metrics': exp_results.get('final_metrics', {})
                                })
                                logger.info(f"♻️ Loaded cached results: {config_name}_seed_{seed}")
                                continue
                            except Exception as e:
                                logger.warning(f"Failed to load cached results for {experiment_name}: {e}")
                    # No cached result found; record as failed in postprocess-only mode
                    completed_experiments.append({
                        'success': False,
                        'config_name': config_name,
                        'seed': seed,
                        'error': 'No cached results found (postprocess_only)'
                    })
        else:
            if parallel_single_endpoints:
                # Group experiments: run all single endpoints in parallel, others sequentially
                single_endpoint_experiments = [(cn, ci, s) for cn, ci, s in experiments if 'single_' in cn]
                other_experiments = [(cn, ci, s) for cn, ci, s in experiments if 'single_' not in cn]
                
                logger.info(f"🚀 Running {len(single_endpoint_experiments)} single endpoints in parallel...")
                logger.info(f"⏳ Then running {len(other_experiments)} other experiments sequentially...")
                
                # Run all single endpoints in parallel
                if single_endpoint_experiments:
                    with ThreadPoolExecutor(max_workers=min(len(single_endpoint_experiments), max_workers)) as executor:
                        future_to_exp = {
                            executor.submit(self.run_single_experiment, config_name, config_info, seed): (config_name, seed)
                            for config_name, config_info, seed in single_endpoint_experiments
                        }
                        
                        for future in as_completed(future_to_exp):
                            config_name, seed = future_to_exp[future]
                            try:
                                result = future.result()
                                completed_experiments.append(result)
                                
                                if result['success']:
                                    logger.info(f"✅ {config_name}_seed_{seed}: {result['training_time']:.1f}s")
                                else:
                                    logger.warning(f"❌ {config_name}_seed_{seed}: {result['error']}")
                                    
                            except Exception as e:
                                logger.error(f"💥 {config_name}_seed_{seed}: Exception: {e}")
                
                # Run other experiments sequentially
                for config_name, config_info, seed in other_experiments:
                    result = self.run_single_experiment(config_name, config_info, seed)
                    completed_experiments.append(result)
                    
                    if result['success']:
                        logger.info(f"✅ {config_name}_seed_{seed}: {result['training_time']:.1f}s")
                    else:
                        logger.warning(f"❌ {config_name}_seed_{seed}: {result['error']}")
            else:
                # Original sequential/parallel execution
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all experiments
                    future_to_exp = {
                        executor.submit(self.run_single_experiment, config_name, config_info, seed): (config_name, seed)
                        for config_name, config_info, seed in experiments
                    }
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_exp):
                        config_name, seed = future_to_exp[future]
                        try:
                            result = future.result()
                            completed_experiments.append(result)
                            
                            if result['success']:
                                logger.info(f"✅ {config_name}_seed_{seed}: {result['training_time']:.1f}s")
                            else:
                                logger.warning(f"❌ {config_name}_seed_{seed}: {result['error']}")
                                
                        except Exception as e:
                            logger.error(f"💥 {config_name}_seed_{seed}: Exception: {e}")
        
        # Organize results by configuration
        self.results = {}
        for exp in completed_experiments:
            config_name = exp['config_name']
            if config_name not in self.results:
                self.results[config_name] = {
                    'description': configs[config_name]['description'],
                    'task_mode': configs[config_name]['task_mode'],
                    'successful_runs': [],
                    'failed_runs': []
                }
            
            if exp['success']:
                self.results[config_name]['successful_runs'].append(exp)
            else:
                self.results[config_name]['failed_runs'].append(exp)
        
        logger.info(f"\n📊 Completed {len(completed_experiments)} experiments")
        for config_name, data in self.results.items():
            n_success = len(data['successful_runs'])
            n_total = n_success + len(data['failed_runs'])
            logger.info(f"  {config_name}: {n_success}/{n_total} successful")
    
    def compute_statistics(self):
        """Compute comprehensive statistical analysis."""
        
        logger.info("📈 Computing sensitivity analysis statistics...")
        
        self.comparison_stats = {}
        
        for config_name, data in self.results.items():
            if not data['successful_runs']:
                continue
                
            # Extract metrics from successful runs
            metrics_data = {}
            for run in data['successful_runs']:
                final_metrics = run.get('final_metrics', {})
                
                # Extract key metrics
                cls_aucs = [v for k, v in final_metrics.items() if 'auc' in k.lower()]
                cls_accs = [v for k, v in final_metrics.items() if 'accuracy' in k.lower()]
                reg_r2s = [v for k, v in final_metrics.items() if 'r2' in k.lower()]
                reg_rmses = [v for k, v in final_metrics.items() if 'rmse' in k.lower()]
                
                if cls_aucs:
                    metrics_data.setdefault('avg_auc', []).append(np.mean(cls_aucs))
                if cls_accs:
                    metrics_data.setdefault('avg_accuracy', []).append(np.mean(cls_accs))
                if reg_r2s:
                    metrics_data.setdefault('avg_r2', []).append(np.mean(reg_r2s))
                if reg_rmses:
                    metrics_data.setdefault('avg_rmse', []).append(np.mean(reg_rmses))
                
                metrics_data.setdefault('total_loss', []).append(
                    run['results'].get('final_loss', 0)
                )
            
            # Compute statistics for each metric
            stats_summary = {}
            for metric_name, values in metrics_data.items():
                if values:
                    stats_summary[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        'median': float(np.median(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'n_runs': len(values),
                        'raw_values': [float(v) for v in values]
                    }
            
            self.comparison_stats[config_name] = {
                'description': data['description'],
                'task_mode': data['task_mode'],
                'statistics': stats_summary,
                'n_successful_runs': len(data['successful_runs']),
                'n_total_runs': len(data['successful_runs']) + len(data['failed_runs'])
            }
    
    def perform_statistical_tests(self):
        """Perform statistical significance tests."""
        
        logger.info("🧮 Performing statistical significance tests...")
        
        # Use multi_task_all as baseline for comparisons
        baseline_key = 'multi_task_all'
        # Ensure attribute exists even if we early-return
        self.statistical_tests = {}
        if baseline_key not in self.comparison_stats:
            logger.warning("Baseline configuration 'multi_task_all' not found")
            return
        
        baseline_stats = self.comparison_stats[baseline_key]['statistics']
        
        # Reinitialize before filling
        self.statistical_tests = {}
        
        for config_name, data in self.comparison_stats.items():
            if config_name == baseline_key:
                continue
                
            config_stats = data['statistics']
            self.statistical_tests[config_name] = {
                'description': data['description'],
                'comparisons': {}
            }
            
            # Compare each metric
            for metric in ['avg_auc', 'avg_r2', 'avg_accuracy', 'avg_rmse']:
                if metric in baseline_stats and metric in config_stats:
                    baseline_values = baseline_stats[metric]['raw_values']
                    config_values = config_stats[metric]['raw_values']
                    
                    # Perform t-test if we have enough data
                    if len(baseline_values) >= 2 and len(config_values) >= 2:
                        try:
                            # Two-sample t-test
                            t_stat, p_value = stats.ttest_ind(baseline_values, config_values)
                            
                            # Effect size (Cohen's d)
                            pooled_std = np.sqrt(
                                ((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) +
                                 (len(config_values) - 1) * np.var(config_values, ddof=1)) /
                                (len(baseline_values) + len(config_values) - 2)
                            )
                            cohens_d = (np.mean(config_values) - np.mean(baseline_values)) / pooled_std
                            
                            # Determine significance
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                            
                            self.statistical_tests[config_name]['comparisons'][metric] = {
                                'baseline_mean': baseline_stats[metric]['mean'],
                                'config_mean': config_stats[metric]['mean'],
                                'difference': config_stats[metric]['mean'] - baseline_stats[metric]['mean'],
                                'relative_change': (config_stats[metric]['mean'] - baseline_stats[metric]['mean']) / baseline_stats[metric]['mean'] * 100 if baseline_stats[metric]['mean'] != 0 else 0,
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'cohens_d': float(cohens_d),
                                'significance': significance,
                                'interpretation': self._interpret_effect_size(cohens_d)
                            }
                        except Exception as e:
                            logger.warning(f"Statistical test failed for {config_name} {metric}: {e}")
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def create_visualizations(self):
        """Create Figure 6 and other visualizations."""
        
        logger.info("📊 Creating visualizations...")
        
        # Extract data for visualization
        config_names = []
        auc_means = []
        auc_stds = []
        r2_means = []
        r2_stds = []
        
        for config_name, data in self.comparison_stats.items():
            stats = data['statistics']
            
            if 'avg_auc' in stats and stats['avg_auc']['n_runs'] > 0:
                config_names.append(config_name.replace('_', ' ').title())
                auc_means.append(stats['avg_auc']['mean'])
                auc_stds.append(stats['avg_auc']['std'])
                
                # Add R2 if available, otherwise 0
                if 'avg_r2' in stats and stats['avg_r2']['n_runs'] > 0:
                    r2_means.append(stats['avg_r2']['mean'])
                    r2_stds.append(stats['avg_r2']['std'])
                else:
                    r2_means.append(0.0)
                    r2_stds.append(0.0)
        
        if len(config_names) > 0:
            # Figure 6: Paired line plot (AUC/R²)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # AUC plot
            x_pos = np.arange(len(config_names))
            ax1.errorbar(x_pos, auc_means, yerr=auc_stds, marker='o', linewidth=2, markersize=6)
            ax1.set_xlabel('Configuration')
            ax1.set_ylabel('Average AUC')
            ax1.set_title('Classification Performance (AUC)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(config_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # R² plot
            ax2.errorbar(x_pos, r2_means, yerr=r2_stds, marker='s', linewidth=2, markersize=6, color='orange')
            ax2.set_xlabel('Configuration')
            ax2.set_ylabel('Average R²')
            ax2.set_title('Regression Performance (R²)')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(config_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_path = self.output_dir / "Figure_6_sensitivity_analysis.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight')
            logger.info(f"📊 Figure 6 saved to: {fig_path}")
            plt.close()
    
    def generate_table_s4(self):
        """Generate Table S.A4 with actual experimental results."""
        
        logger.info("📋 Generating Table S.A4...")
        
        # Prepare table data
        table_data = []
        
        baseline_key = 'multi_task_all'
        baseline_auc = self.comparison_stats.get(baseline_key, {}).get('statistics', {}).get('avg_auc', {}).get('mean', 0)
        baseline_r2 = self.comparison_stats.get(baseline_key, {}).get('statistics', {}).get('avg_r2', {}).get('mean', 0)
        
        for config_name, data in self.comparison_stats.items():
            stats = data['statistics']
            
            # Extract metrics
            auc_mean = stats.get('avg_auc', {}).get('mean', 0)
            auc_std = stats.get('avg_auc', {}).get('std', 0)
            r2_mean = stats.get('avg_r2', {}).get('mean', 0)
            r2_std = stats.get('avg_r2', {}).get('std', 0)
            
            # Calculate relative performance
            if config_name == baseline_key:
                relative_auc = 0.0
                relative_r2 = 0.0
            else:
                relative_auc = ((auc_mean - baseline_auc) / baseline_auc * 100) if baseline_auc > 0 else 0
                relative_r2 = ((r2_mean - baseline_r2) / baseline_r2 * 100) if baseline_r2 > 0 else 0
            
            # Get statistical significance (robust if baseline tests missing)
            significance = "ns"
            p_value = 1.0
            stat_tests = getattr(self, 'statistical_tests', {})
            if config_name in stat_tests:
                auc_comparison = stat_tests[config_name]['comparisons'].get('avg_auc', {})
                significance = auc_comparison.get('significance', 'ns')
                p_value = auc_comparison.get('p_value', 1.0)
            
            # Performance change interpretation
            overall_change = (relative_auc + relative_r2) / 2
            if abs(overall_change) < 1:
                interpretation = "No significant difference"
            elif overall_change > 5:
                interpretation = "Significant improvement"
            elif overall_change < -5:
                interpretation = "Significant degradation"
            else:
                interpretation = "Marginal difference"
            
            table_data.append({
                'Configuration': data['description'],
                'Classification AUC': f"{auc_mean:.3f} ± {auc_std:.3f}",
                'Regression R²': f"{r2_mean:.3f} ± {r2_std:.3f}",
                'Overall Performance': f"{(auc_mean + r2_mean)/2:.3f}",
                'Relative to Baseline (%)': f"{overall_change:.1f}%",
                'Performance Change': "improvement" if overall_change > 0 else "degradation" if overall_change < 0 else "baseline",
                'Statistical Significance': significance,
                'p-value': f"{p_value:.3f}",
                'Interpretation': interpretation
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(table_data)
        
        # Sort by overall performance (descending)
        df['_sort_key'] = df['Overall Performance'].str.extract(r'(\d+\.\d+)').astype(float)
        df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)
        
        # Save table
        table_path = self.output_dir / "Table_S4_sensitivity_analysis.csv"
        df.to_csv(table_path, index=False)
        
        # Also save as formatted CSV for LaTeX
        latex_path = self.output_dir / "Table_S4_latex_format.tex"
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, escape=False))
        
        logger.info(f"📋 Table S.A4 saved to: {table_path}")
        logger.info(f"📋 LaTeX version saved to: {latex_path}")
    
    def save_comprehensive_results(self):
        """Save all results and analysis."""
        
        logger.info("💾 Saving comprehensive results...")
        
        final_results = {
            'experiment_info': {
                'type': 'R1C3_sensitivity_analysis',
                'description': 'Single endpoints vs aggregated scores comparison',
                'timestamp': datetime.now().isoformat(),
                'total_experiments': sum(len(data['successful_runs']) + len(data['failed_runs']) 
                                       for data in self.results.values())
            },
            'base_configuration': self.base_config,
            'experiment_results': self.results,
            'statistical_summary': self.comparison_stats,
            'statistical_tests': getattr(self, 'statistical_tests', {}),
            'key_findings': self._generate_key_findings()
        }
        
        results_file = self.output_dir / "comprehensive_sensitivity_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"💾 Comprehensive results saved to: {results_file}")
    
    def _generate_key_findings(self) -> Dict:
        """Generate key findings summary."""
        
        findings = {
            'total_configurations_tested': len(self.comparison_stats),
            'successful_baseline_runs': 0,
            'best_single_endpoint_performance': {},
            'multi_task_vs_single_task_summary': {}
        }
        
        # Find baseline performance
        baseline_key = 'multi_task_all'
        if baseline_key in self.comparison_stats:
            findings['successful_baseline_runs'] = self.comparison_stats[baseline_key]['n_successful_runs']
        
        # Find best performing single endpoint
        best_single_auc = 0
        best_single_config = None
        
        for config_name, data in self.comparison_stats.items():
            if config_name.startswith('single_'):
                stats = data['statistics']
                if 'avg_auc' in stats and stats['avg_auc']['mean'] > best_single_auc:
                    best_single_auc = stats['avg_auc']['mean']
                    best_single_config = config_name
        
        if best_single_config:
            findings['best_single_endpoint_performance'] = {
                'configuration': best_single_config,
                'description': self.comparison_stats[best_single_config]['description'],
                'auc': best_single_auc
            }
        
        return findings
    
    def print_summary(self):
        """Print executive summary."""
        
        logger.info("\n🎉 R1C3 SENSITIVITY ANALYSIS SUMMARY")
        logger.info("="*60)
        
        # Overall statistics
        total_configs = len(self.comparison_stats)
        total_successful = sum(data['n_successful_runs'] for data in self.comparison_stats.values())
        
        logger.info(f"📊 Tested {total_configs} configurations")
        logger.info(f"✅ {total_successful} successful experiments total")
        
        # Key comparisons
        if hasattr(self, 'statistical_tests'):
            logger.info("\n🔍 Key Comparisons vs Multi-task Baseline:")
            
            for config_name, test_data in self.statistical_tests.items():
                if 'avg_auc' in test_data['comparisons']:
                    auc_comp = test_data['comparisons']['avg_auc']
                    logger.info(f"  {test_data['description']}:")
                    logger.info(f"    AUC: {auc_comp['baseline_mean']:.4f} → {auc_comp['config_mean']:.4f}")
                    logger.info(f"    Change: {auc_comp['relative_change']:.1f}% ({auc_comp['significance']})")
        
        logger.info(f"\n📁 Results saved to: {self.output_dir}")
        logger.info("🎊 Analysis completed successfully!")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='R1C3 Sensitivity Analysis')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                       help='Random seeds for experiments')
    parser.add_argument('--max_workers', type=int, default=2,
                       help='Maximum parallel workers')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--skip_multitask', action='store_true',
                       help='Skip multi-task baseline training (use existing results)')
    parser.add_argument('--single_endpoints_only', action='store_true',
                       help='Run only single endpoint experiments (fastest option)')
    parser.add_argument('--parallel_single_endpoints', action='store_true',
                       help='Run all single endpoints in parallel (much faster if enough GPU memory)')
    parser.add_argument('--postprocess_only', action='store_true',
                       help='Skip all training; only load existing results and analyze')

    # Head-only fine-tuning options (shared trunk from multi_full)
    parser.add_argument('--head_only_ft', action='store_true',
                       help='Enable head-only fine-tuning for single endpoints using a shared trunk')
    parser.add_argument('--resume_from_ckpt', type=str, default=None,
                       help='Path to multi_full best checkpoint (.pth) used as shared trunk')
    parser.add_argument('--head_ft_epochs', type=int, default=3,
                       help='Epochs for head-only fine-tuning')
    parser.add_argument('--head_ft_lr', type=float, default=5e-4,
                       help='Learning rate for head-only fine-tuning')
    parser.add_argument('--head_ft_warmup', type=float, default=0.02,
                       help='Warmup ratio for head-only fine-tuning')
    parser.add_argument('--cover_all_cls', action='store_true',
                       help='Cover all 26 classification endpoints for single-task runs')
    parser.add_argument('--cover_all_reg', action='store_true',
                       help='Cover all 5 regression endpoints for single-task runs')
    parser.add_argument('--ingest_baseline_results', type=str, default=None,
                       help='Path to a multi_full results.json to use as baseline without retraining')
    parser.add_argument('--baseline_name', type=str, default='multi_task_all',
                       help='Key name for ingested baseline (default: multi_task_all)')
    
    args = parser.parse_args()
    
    # Base configuration for all experiments
    base_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': 1e-4,
        'use_preprocessed': True,
        'preprocessed_dir': 'data/data/processed',
        'deterministic': True,
        'warmup_ratio': 0.06
    }
    
    # Run analysis
    analyzer = SensitivityAnalyzer(base_config)
    
    try:
        # Use optimized settings
        skip_multitask = args.skip_multitask or args.single_endpoints_only
        
        analyzer.run_sensitivity_analysis(
            seeds=args.seeds, 
            max_workers=args.max_workers, 
            skip_multitask=skip_multitask,
            parallel_single_endpoints=args.parallel_single_endpoints,
            postprocess_only=args.postprocess_only,
            cover_all_cls=args.cover_all_cls,
            cover_all_reg=args.cover_all_reg,
            head_only_ft=args.head_only_ft,
            resume_from_ckpt=args.resume_from_ckpt,
            head_ft_epochs=args.head_ft_epochs,
            head_ft_lr=args.head_ft_lr,
            head_ft_warmup=args.head_ft_warmup,
        )
        # Ingest existing multi_full baseline results (without retraining)
        if args.ingest_baseline_results and Path(args.ingest_baseline_results).exists():
            analyzer.ingest_baseline_results(
                results_json_path=args.ingest_baseline_results,
                config_name=args.baseline_name,
                seed=args.seeds[0] if isinstance(args.seeds, list) and len(args.seeds) > 0 else 42,
            )
        analyzer.compute_statistics()
        analyzer.perform_statistical_tests()
        analyzer.create_visualizations()
        analyzer.generate_table_s4()
        # Export per-endpoint delta CSV if baseline exists
        analyzer.export_single_vs_multi_delta(
            analyzer.output_dir / 'single_vs_multi_delta.csv',
            baseline_key=args.baseline_name
        )
        analyzer.save_comprehensive_results()
        analyzer.print_summary()
        
        logger.info("🎉 R1C3 sensitivity analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
