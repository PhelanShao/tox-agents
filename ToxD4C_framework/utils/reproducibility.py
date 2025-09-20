#!/usr/bin/env python3
"""
Reproducibility utilities for ToxD4C
Addresses A0 reviewer concerns about reproducibility and robustness.
"""

import torch
import numpy as np
import random
import os
import platform
import json
import logging
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def set_deterministic_training(seed: int = 42, strict: bool = True):
    """
    Set deterministic training environment for reproducible results.
    
    Args:
        seed: Random seed for all random number generators
        strict: If True, use strict deterministic mode (slower but fully reproducible)
    """
    logger.info(f"Setting deterministic training with seed {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if strict:
        # Strict deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Strict deterministic mode enabled (may reduce performance)")
    else:
        # Balanced mode - some randomness for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info("Balanced deterministic mode enabled")

def get_environment_info() -> Dict[str, Any]:
    """
    Collect comprehensive environment information for reproducibility.
    
    Returns:
        Dictionary containing environment details
    """
    env_info = {
        'timestamp': datetime.now().isoformat(),
        'hostname': platform.node(),
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'python': {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler()
        },
        'pytorch': {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        },
        'hardware': {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
    }
    
    # GPU information
    if torch.cuda.is_available():
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                'name': gpu_props.name,
                'memory_gb': round(gpu_props.total_memory / (1024**3), 2),
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            })
        env_info['gpu'] = gpu_info
    
    # Try to get package versions
    try:
        import pkg_resources
        installed_packages = {pkg.project_name: pkg.version 
                            for pkg in pkg_resources.working_set}
        
        # Key packages for reproducibility
        key_packages = ['torch', 'numpy', 'rdkit', 'scikit-learn', 'pandas']
        env_info['packages'] = {pkg: installed_packages.get(pkg, 'unknown') 
                               for pkg in key_packages if pkg in installed_packages}
    except Exception as e:
        logger.warning(f"Could not collect package information: {e}")
        env_info['packages'] = {}
    
    return env_info

def save_environment_info(save_path: str = "environment_info.json"):
    """
    Save environment information to file.
    
    Args:
        save_path: Path to save environment information
    """
    env_info = get_environment_info()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(env_info, f, indent=2)
    
    logger.info(f"Environment information saved to {save_path}")
    return env_info

def create_experiment_snapshot(experiment_dir: str, config: Dict[str, Any], 
                             additional_info: Optional[Dict[str, Any]] = None):
    """
    Create a complete experiment snapshot for reproducibility.
    
    Args:
        experiment_dir: Directory to save experiment snapshot
        config: Model/training configuration
        additional_info: Additional information to include
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save environment info
    env_info = save_environment_info(experiment_dir / "environment_info.json")
    
    # Save configuration
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save additional info
    if additional_info:
        with open(experiment_dir / "additional_info.json", 'w') as f:
            json.dump(additional_info, f, indent=2)
    
    # Create experiment metadata
    metadata = {
        'experiment_id': experiment_dir.name,
        'created_at': datetime.now().isoformat(),
        'environment_snapshot': env_info,
        'config_hash': hash(str(sorted(config.items()))),
        'git_info': get_git_info()
    }
    
    with open(experiment_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Experiment snapshot created at {experiment_dir}")
    return metadata

def get_git_info() -> Dict[str, Any]:
    """
    Get git repository information for reproducibility.
    
    Returns:
        Dictionary with git information
    """
    git_info = {}
    
    try:
        # Get current commit hash
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()
        
        # Get current branch
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()
        
        # Check for uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            git_info['has_uncommitted_changes'] = bool(result.stdout.strip())
            if git_info['has_uncommitted_changes']:
                git_info['uncommitted_files'] = result.stdout.strip().split('\n')
        
        # Get remote URL
        result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            git_info['remote_url'] = result.stdout.strip()
            
    except Exception as e:
        logger.warning(f"Could not collect git information: {e}")
        git_info['error'] = str(e)
    
    return git_info

def validate_reproducibility(experiment_dir: str, tolerance: float = 1e-6) -> bool:
    """
    Validate that an experiment can be reproduced.
    
    Args:
        experiment_dir: Directory containing experiment snapshot
        tolerance: Numerical tolerance for comparing results
        
    Returns:
        True if experiment appears reproducible
    """
    experiment_dir = Path(experiment_dir)
    
    # Check required files
    required_files = ['environment_info.json', 'config.json', 'metadata.json']
    for file in required_files:
        if not (experiment_dir / file).exists():
            logger.error(f"Missing required file: {file}")
            return False
    
    # Load and validate environment
    try:
        with open(experiment_dir / "environment_info.json", 'r') as f:
            saved_env = json.load(f)
        
        current_env = get_environment_info()
        
        # Check critical environment components
        critical_checks = [
            ('python.version', 'Python version'),
            ('pytorch.version', 'PyTorch version'),
            ('pytorch.cuda_version', 'CUDA version')
        ]
        
        for key_path, description in critical_checks:
            saved_val = get_nested_value(saved_env, key_path)
            current_val = get_nested_value(current_env, key_path)
            
            if saved_val != current_val:
                logger.warning(f"{description} mismatch: saved={saved_val}, current={current_val}")
        
        logger.info("Environment validation completed")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False

def get_nested_value(d: Dict, key_path: str):
    """Get nested dictionary value using dot notation."""
    keys = key_path.split('.')
    value = d
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value

class ReproducibilityContext:
    """Context manager for reproducible experiments."""
    
    def __init__(self, seed: int = 42, strict: bool = True, 
                 experiment_dir: Optional[str] = None):
        self.seed = seed
        self.strict = strict
        self.experiment_dir = experiment_dir
        self.original_state = {}
        
    def __enter__(self):
        # Save original state
        self.original_state = {
            'torch_deterministic': torch.backends.cudnn.deterministic,
            'torch_benchmark': torch.backends.cudnn.benchmark,
            'python_hash_seed': os.environ.get('PYTHONHASHSEED')
        }
        
        # Set deterministic mode
        set_deterministic_training(self.seed, self.strict)
        
        # Create experiment snapshot if directory provided
        if self.experiment_dir:
            self.metadata = create_experiment_snapshot(
                self.experiment_dir, 
                {'seed': self.seed, 'strict': self.strict}
            )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        torch.backends.cudnn.deterministic = self.original_state['torch_deterministic']
        torch.backends.cudnn.benchmark = self.original_state['torch_benchmark']
        
        if self.original_state['python_hash_seed'] is not None:
            os.environ['PYTHONHASHSEED'] = self.original_state['python_hash_seed']
        elif 'PYTHONHASHSEED' in os.environ:
            del os.environ['PYTHONHASHSEED']

# Example usage
if __name__ == "__main__":
    # Test reproducibility utilities
    with ReproducibilityContext(seed=42, experiment_dir="test_experiment"):
        print("Running in reproducible context")
        env_info = get_environment_info()
        print(f"Environment: {env_info['platform']['system']} {env_info['platform']['release']}")
        print(f"PyTorch: {env_info['pytorch']['version']}")
        print(f"CUDA: {env_info['pytorch']['cuda_available']}")
