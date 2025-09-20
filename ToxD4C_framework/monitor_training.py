#!/usr/bin/env python3
"""
Training monitoring script for ToxD4C experiments
Monitors multiple training processes and reports progress.
"""

import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_status():
    """Check GPU memory usage and running processes."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_used, memory_total, gpu_util = result.stdout.strip().split(', ')
            return {
                'memory_used_mb': int(memory_used),
                'memory_total_mb': int(memory_total),
                'memory_usage_percent': round(int(memory_used) / int(memory_total) * 100, 1),
                'gpu_utilization_percent': int(gpu_util)
            }
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
    return None

def check_experiment_progress():
    """Check progress of running experiments."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return []
    
    experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            # Look for training logs or checkpoints
            log_files = list(exp_dir.glob("*.log"))
            checkpoint_dir = exp_dir / "checkpoints"
            
            exp_info = {
                'name': exp_dir.name,
                'start_time': datetime.fromtimestamp(exp_dir.stat().st_ctime).isoformat(),
                'status': 'unknown',
                'latest_epoch': None,
                'latest_loss': None,
                'checkpoints': 0
            }
            
            # Check for checkpoints
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.pth"))
                exp_info['checkpoints'] = len(checkpoints)
                if checkpoints:
                    exp_info['status'] = 'running'
            
            # Parse log files for latest progress
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-50:]):  # Check last 50 lines
                            if 'Epoch' in line and 'Loss=' in line:
                                # Extract epoch and loss info
                                if 'Epoch' in line and '/' in line:
                                    epoch_part = line.split('Epoch')[1].split('/')[0].strip()
                                    try:
                                        exp_info['latest_epoch'] = int(epoch_part)
                                    except:
                                        pass
                                if 'Loss=' in line:
                                    loss_part = line.split('Loss=')[1].split(',')[0].strip()
                                    try:
                                        exp_info['latest_loss'] = float(loss_part)
                                    except:
                                        pass
                                break
                except Exception as e:
                    logger.warning(f"Error reading log file {log_file}: {e}")
            
            experiments.append(exp_info)
    
    return experiments

def get_process_info():
    """Get information about running Python processes."""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if result.returncode == 0:
            python_processes = []
            for line in result.stdout.split('\n'):
                if 'python' in line and 'train.py' in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        pid = parts[1]
                        cpu_percent = parts[2]
                        mem_percent = parts[3]
                        command = ' '.join(parts[10:])
                        
                        # Extract experiment name
                        exp_name = 'unknown'
                        if '--experiment_name' in command:
                            try:
                                idx = command.split().index('--experiment_name')
                                if idx + 1 < len(command.split()):
                                    exp_name = command.split()[idx + 1]
                            except:
                                pass
                        
                        python_processes.append({
                            'pid': pid,
                            'cpu_percent': cpu_percent,
                            'mem_percent': mem_percent,
                            'experiment_name': exp_name,
                            'command': command[:100] + '...' if len(command) > 100 else command
                        })
            return python_processes
    except Exception as e:
        logger.error(f"Failed to get process info: {e}")
    return []

def print_status_report():
    """Print comprehensive status report."""
    print("\n" + "="*80)
    print(f"🔬 ToxD4C Training Status Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # GPU Status
    gpu_status = check_gpu_status()
    if gpu_status:
        print(f"🖥️  GPU Status:")
        print(f"   Memory: {gpu_status['memory_used_mb']:,} MB / {gpu_status['memory_total_mb']:,} MB ({gpu_status['memory_usage_percent']}%)")
        print(f"   Utilization: {gpu_status['gpu_utilization_percent']}%")
    else:
        print("🖥️  GPU Status: Unable to retrieve")
    
    # Running Processes
    processes = get_process_info()
    if processes:
        print(f"\n🚀 Running Training Processes ({len(processes)}):")
        for proc in processes:
            print(f"   PID {proc['pid']}: {proc['experiment_name']}")
            print(f"      CPU: {proc['cpu_percent']}%, Memory: {proc['mem_percent']}%")
    else:
        print("\n🚀 Running Training Processes: None detected")
    
    # Experiment Progress
    experiments = check_experiment_progress()
    if experiments:
        print(f"\n📊 Experiment Progress ({len(experiments)}):")
        for exp in experiments:
            print(f"   📁 {exp['name']}")
            print(f"      Status: {exp['status']}")
            print(f"      Started: {exp['start_time']}")
            if exp['latest_epoch']:
                print(f"      Latest Epoch: {exp['latest_epoch']}")
            if exp['latest_loss']:
                print(f"      Latest Loss: {exp['latest_loss']:.4f}")
            print(f"      Checkpoints: {exp['checkpoints']}")
    else:
        print("\n📊 Experiment Progress: No experiments found")
    
    print("\n" + "="*80)

def monitor_training(interval_minutes=5, max_duration_hours=12):
    """Monitor training progress continuously."""
    logger.info(f"🔍 Starting training monitor (interval: {interval_minutes} min, max duration: {max_duration_hours} hours)")
    
    start_time = time.time()
    max_duration_seconds = max_duration_hours * 3600
    interval_seconds = interval_minutes * 60
    
    try:
        while True:
            print_status_report()
            
            # Check if we should stop monitoring
            elapsed_time = time.time() - start_time
            if elapsed_time > max_duration_seconds:
                logger.info(f"⏰ Maximum monitoring duration ({max_duration_hours} hours) reached")
                break
            
            # Check if any processes are still running
            processes = get_process_info()
            if not processes:
                logger.info("✅ No training processes detected. Monitoring complete.")
                break
            
            # Wait for next check
            logger.info(f"⏳ Next check in {interval_minutes} minutes...")
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        logger.info("🛑 Monitoring stopped by user")
    except Exception as e:
        logger.error(f"💥 Monitoring error: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor ToxD4C training progress')
    parser.add_argument('--interval', type=int, default=5, 
                       help='Check interval in minutes (default: 5)')
    parser.add_argument('--max_hours', type=int, default=12,
                       help='Maximum monitoring duration in hours (default: 12)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    
    args = parser.parse_args()
    
    if args.once:
        print_status_report()
    else:
        monitor_training(interval_minutes=args.interval, max_duration_hours=args.max_hours)

if __name__ == "__main__":
    main()
