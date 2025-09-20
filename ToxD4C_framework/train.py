import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent))

from data.lmdb_dataset import create_lmdb_dataloaders
from models.toxd4c import ToxD4C
from configs.toxd4c_config import get_enhanced_toxd4c_config

# Import new utilities for addressing reviewer concerns
from utils.reproducibility import ReproducibilityContext, save_environment_info, create_experiment_snapshot
from utils.splitter import MolecularSplitter, create_external_validation_splits
from utils.uncertainty import TemperatureScaling, DeepEnsemble, ApplicabilityDomain, ConformalPrediction

# Import preprocessing functionality
from preprocess_data import preprocess_lmdb

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def check_for_nan_inf(tensor, name="tensor"):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan:
        logger.warning(f"{name} contains NaN values!")
    if has_inf:
        logger.warning(f"{name} contains Inf values!")
    
    return has_nan or has_inf


def safe_loss_computation(pred, target, mask, loss_fn):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    valid_pred = pred[mask]
    valid_target = target[mask]
    
    if check_for_nan_inf(valid_pred, "prediction") or check_for_nan_inf(valid_target, "target"):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    loss = loss_fn(valid_pred, valid_target)
    
    if check_for_nan_inf(loss, "loss"):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    return loss


def compute_metrics(predictions, targets, masks, task_type='classification'):
    metrics = {}
    
    for task_idx in range(predictions.shape[1]):
        task_pred = predictions[:, task_idx]
        task_target = targets[:, task_idx]
        task_mask = masks[:, task_idx]
        
        if task_mask.sum() == 0:
            continue
        
        valid_pred = task_pred[task_mask].cpu().numpy()
        valid_target = task_target[task_mask].cpu().numpy()
        
        if len(valid_pred) == 0:
            continue
        
        try:
            if task_type == 'classification':
                pred_binary = (valid_pred > 0.5).astype(int)
                target_binary = valid_target.astype(int)
                
                acc = accuracy_score(target_binary, pred_binary)
                metrics[f'task_{task_idx}_accuracy'] = acc
                
                if len(np.unique(target_binary)) > 1:
                    auc = roc_auc_score(target_binary, valid_pred)
                    metrics[f'task_{task_idx}_auc'] = auc
            else:
                mse = mean_squared_error(valid_target, valid_pred)
                if np.var(valid_target) < 1e-6:
                    r2 = 0.0
                else:
                    r2 = r2_score(valid_target, valid_pred)
                rmse = np.sqrt(mse)
                
                metrics[f'task_{task_idx}_mse'] = mse
                metrics[f'task_{task_idx}_rmse'] = rmse
                metrics[f'task_{task_idx}_r2'] = r2
        except Exception as e:
            logger.warning(f"Error computing metrics for task {task_idx}: {e}")
            continue
    
    return metrics


def train_epoch(model, dataloader, optimizer, scheduler, device, config=None):
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    classification_criterion = nn.BCEWithLogitsLoss(reduction='none')
    regression_criterion = nn.MSELoss(reduction='none')
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            data = {
                'atom_features': batch['atom_features'].to(device),
                'edge_index': batch['edge_index'].to(device),
                'coordinates': batch['coordinates'].to(device),
                'batch': batch['batch'].to(device)
            }
            
            cls_labels = batch['classification_labels'].to(device)
            reg_labels = batch['regression_labels'].to(device)
            cls_mask = batch['classification_mask'].to(device)
            reg_mask = batch['regression_mask'].to(device)
            
            # Adjust labels and masks for single endpoint mode
            if config.get('task_mode') == 'single':
                if config.get('single_endpoint_cls') is not None:
                    endpoint_idx = config['single_endpoint_cls']
                    cls_labels = cls_labels[:, endpoint_idx:endpoint_idx+1]
                    cls_mask = cls_mask[:, endpoint_idx:endpoint_idx+1]
                elif config.get('single_endpoint_reg') is not None:
                    endpoint_idx = config['single_endpoint_reg']
                    reg_labels = reg_labels[:, endpoint_idx:endpoint_idx+1]
                    reg_mask = reg_mask[:, endpoint_idx:endpoint_idx+1]
            smiles_list = batch['smiles']
            
            optimizer.zero_grad()
            
            outputs = model(data, smiles_list)
            
            cls_preds = outputs['predictions']['classification']
            reg_preds = outputs['predictions']['regression']

            # Determine which branches are active by output shape
            classification_active = cls_preds.numel() > 0
            regression_active = reg_preds.numel() > 0

            if classification_active and check_for_nan_inf(cls_preds, "classification_output"):
                logger.warning(f"NaN in classification output at batch {batch_idx}")
                continue
            if regression_active and check_for_nan_inf(reg_preds, "regression_output"):
                logger.warning(f"NaN in regression output at batch {batch_idx}")
                continue

            # Compute losses only for active branches; otherwise zero loss
            if classification_active:
                cls_loss = safe_loss_computation(
                    cls_preds, cls_labels, cls_mask,
                    lambda p, t: classification_criterion(p, t).mean()
                )
            else:
                cls_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if regression_active:
                reg_loss = safe_loss_computation(
                    reg_preds, reg_labels, reg_mask,
                    lambda p, t: regression_criterion(p, t).mean()
                )
            else:
                reg_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Optional: supervised contrastive learning on graph representations
            contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if hasattr(model, 'contrastive_loss') and 'contrastive_features' in outputs:
                try:
                    contrastive_features = outputs['contrastive_features']
                    # Build label vectors for contrastive supervision
                    contrastive_labels = torch.cat([cls_labels, reg_labels], dim=1)
                    contrastive_loss = model.compute_contrastive_loss(
                        contrastive_features, contrastive_labels
                    )
                except Exception as e:
                    logger.warning(f"Contrastive loss computation failed at batch {batch_idx}: {e}")
                    contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)

            c_weight = getattr(getattr(model, 'config', {}), 'get', lambda k, d=None: d)('contrastive_weight', 0.0) if hasattr(model, 'config') else 0.0
            try:
                # model.config is a dict; safer to access directly
                if hasattr(model, 'config') and isinstance(model.config, dict):
                    c_weight = model.config.get('contrastive_weight', 0.0) if hasattr(model, 'contrastive_loss') else 0.0
            except Exception:
                pass

            total_loss_batch = cls_loss + reg_loss + (c_weight * contrastive_loss)
            
            if check_for_nan_inf(total_loss_batch, "total_loss"):
                logger.warning(f"NaN in total loss at batch {batch_idx}")
                continue
            
            total_loss_batch.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and check_for_nan_inf(param.grad, f"grad_{name}"):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                logger.warning(f"NaN in gradients at batch {batch_idx}")
                continue
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}: "
                          f"Loss={total_loss_batch.item():.4f}, "
                          f"Cls={cls_loss.item():.4f}, "
                          f"Reg={reg_loss.item():.4f}, "
                          f"Con={contrastive_loss.item() if hasattr(model, 'contrastive_loss') else 0.0:.4f}")
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    
    return total_loss / num_batches, total_cls_loss / num_batches, total_reg_loss / num_batches


def evaluate_model(model, dataloader, device, config=None):
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    all_cls_preds = []
    all_cls_targets = []
    all_cls_masks = []
    all_reg_preds = []
    all_reg_targets = []
    all_reg_masks = []
    
    classification_criterion = nn.BCEWithLogitsLoss(reduction='none')
    regression_criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                data = {
                    'atom_features': batch['atom_features'].to(device),
                    'edge_index': batch['edge_index'].to(device),
                    'coordinates': batch['coordinates'].to(device),
                    'batch': batch['batch'].to(device)
                }
                
                cls_labels = batch['classification_labels'].to(device)
                reg_labels = batch['regression_labels'].to(device)
                cls_mask = batch['classification_mask'].to(device)
                reg_mask = batch['regression_mask'].to(device)
                
                # Adjust labels and masks for single endpoint mode
                if config.get('task_mode') == 'single':
                    if config.get('single_endpoint_cls') is not None:
                        endpoint_idx = config['single_endpoint_cls']
                        cls_labels = cls_labels[:, endpoint_idx:endpoint_idx+1]
                        cls_mask = cls_mask[:, endpoint_idx:endpoint_idx+1]
                    elif config.get('single_endpoint_reg') is not None:
                        endpoint_idx = config['single_endpoint_reg']
                        reg_labels = reg_labels[:, endpoint_idx:endpoint_idx+1]
                        reg_mask = reg_mask[:, endpoint_idx:endpoint_idx+1]
                smiles_list = batch['smiles']
                
                outputs = model(data, smiles_list)
                
                cls_preds = outputs['predictions']['classification']
                reg_preds = outputs['predictions']['regression']

                classification_active = cls_preds.numel() > 0
                regression_active = reg_preds.numel() > 0

                if classification_active and check_for_nan_inf(cls_preds, "eval_classification"):
                    continue
                if regression_active and check_for_nan_inf(reg_preds, "eval_regression"):
                    continue

                if classification_active:
                    cls_loss = safe_loss_computation(
                        cls_preds, cls_labels, cls_mask,
                        lambda p, t: classification_criterion(p, t).mean()
                    )
                else:
                    cls_loss = torch.tensor(0.0, device=device)

                if regression_active:
                    reg_loss = safe_loss_computation(
                        reg_preds, reg_labels, reg_mask,
                        lambda p, t: regression_criterion(p, t).mean()
                    )
                else:
                    reg_loss = torch.tensor(0.0, device=device)

                total_loss += (cls_loss + reg_loss).item()
                total_cls_loss += cls_loss.item() if classification_active else 0.0
                total_reg_loss += reg_loss.item() if regression_active else 0.0
                num_batches += 1

                if classification_active:
                    cls_probs = torch.sigmoid(cls_preds)
                    all_cls_preds.append(cls_probs.cpu())
                    all_cls_targets.append(cls_labels.cpu())
                    all_cls_masks.append(cls_mask.cpu())

                if regression_active:
                    all_reg_preds.append(reg_preds.cpu())
                    all_reg_targets.append(reg_labels.cpu())
                    all_reg_masks.append(reg_mask.cpu())
                
            except Exception as e:
                logger.error(f"Error in evaluation batch: {e}")
                continue
    
    if num_batches == 0:
        return {}, 0.0, 0.0, 0.0
    
    metrics = {}
    
    if all_cls_preds:
        cls_preds = torch.cat(all_cls_preds, dim=0)
        cls_targets = torch.cat(all_cls_targets, dim=0)
        cls_masks = torch.cat(all_cls_masks, dim=0)
        
        cls_metrics = compute_metrics(cls_preds, cls_targets, cls_masks, 'classification')
        metrics.update(cls_metrics)
    
    if all_reg_preds:
        reg_preds = torch.cat(all_reg_preds, dim=0)
        reg_targets = torch.cat(all_reg_targets, dim=0)
        reg_masks = torch.cat(all_reg_masks, dim=0)
        
        reg_metrics = compute_metrics(reg_preds, reg_targets, reg_masks, 'regression')
        metrics.update(reg_metrics)
    
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    
    return metrics, avg_loss, avg_cls_loss, avg_reg_loss


def main():
    parser = argparse.ArgumentParser(description='ToxD4C Enhanced Training with Reproducibility')
    parser.add_argument('--data_dir', type=str, default='data/dataset', help='Directory for LMDB data')
    parser.add_argument('--experiment_name', type=str, default='toxd4c_enhanced', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_atoms', type=int, default=64, help='Maximum number of atoms')
    parser.add_argument('--warmup_ratio', type=float, default=0.06, help='Learning rate warmup ratio')

    # Reproducibility arguments (A0)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', help='Enable strict deterministic mode')

    # Data splitting arguments (A1)
    parser.add_argument('--split_strategy', type=str, default='random',
                       choices=['random', 'scaffold', 'cluster', 'temporal'],
                       help='Data splitting strategy')
    parser.add_argument('--train_size', type=float, default=0.8, help='Training set fraction')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set fraction')

    # Uncertainty quantification arguments (A2)
    parser.add_argument('--enable_uncertainty', action='store_true',
                       help='Enable uncertainty quantification')
    parser.add_argument('--n_ensemble', type=int, default=3,
                       help='Number of models for ensemble')
    parser.add_argument('--temperature_scaling', action='store_true',
                       help='Enable temperature scaling calibration')

    # External validation arguments
    parser.add_argument('--external_validation', action='store_true',
                       help='Enable external validation with ToxCast/Tox21')
    parser.add_argument('--toxcast_data', type=str, default=None,
                       help='Path to ToxCast data for external validation')
    parser.add_argument('--tox21_data', type=str, default=None,
                       help='Path to Tox21 data for external validation')

    # Preprocessing arguments
    parser.add_argument('--use_preprocessed', action='store_true', default=True,
                       help='Use preprocessed data for faster training')
    parser.add_argument('--preprocessed_dir', type=str, default='data/data/processed',
                       help='Directory containing preprocessed LMDB files')
    parser.add_argument('--force_preprocess', action='store_true',
                       help='Force preprocessing even if preprocessed data exists')

    # Ablation study arguments (A3)
    parser.add_argument('--disable_gnn', action='store_true',
                       help='Disable GNN components for ablation study')
    parser.add_argument('--disable_transformer', action='store_true',
                       help='Disable Transformer components for ablation study')
    parser.add_argument('--disable_geometric', action='store_true',
                       help='Disable geometric encoding for ablation study')
    parser.add_argument('--disable_hierarchical', action='store_true',
                       help='Disable hierarchical encoding for ablation study')
    parser.add_argument('--disable_fingerprint', action='store_true',
                       help='Disable fingerprint branch for ablation study')
    parser.add_argument('--disable_classification', action='store_true',
                       help='Disable classification tasks for ablation study')
    parser.add_argument('--disable_regression', action='store_true',
                       help='Disable regression tasks for ablation study')

    # Contrastive learning toggle (for ablation of SupCon loss)
    parser.add_argument('--disable_contrastive', action='store_true',
                       help='Disable supervised contrastive learning component')
    # Backward-compatible alias used by some runners
    parser.add_argument('--disable_contrastive_loss', action='store_true',
                       help='Alias of --disable_contrastive')

    # Fusion method arguments (for R2C11 cross-attention vs concatenation)
    parser.add_argument('--disable_dynamic_fusion', action='store_true',
                       help='Disable dynamic cross-attention fusion, use concatenation instead')
    parser.add_argument('--fusion_method', type=str, default='cross_attention',
                       choices=['cross_attention', 'concatenation'],
                       help='Fusion method for GNN-Transformer hybrid')

    # GNN backbone selection (for stronger GNN baselines)
    parser.add_argument('--gnn_backbone', type=str, default='default',
                       choices=['default', 'pyg_gcn_stack'],
                       help='Choose GNN backbone: default (current) or PyG GCNConv stack with residual+norm')
    parser.add_argument('--gcn_stack_layers', type=int, default=3,
                       help='Layers for PyG GCNConv stack (recommend 2-4)')

    # Sensitivity analysis arguments (R1C3)
    parser.add_argument('--task_mode', type=str, default='multi',
                       choices=['single', 'multi', 'aggregated'],
                       help='Task mode: single endpoint, multi-task, or aggregated scores')
    parser.add_argument('--single_endpoint_cls', type=int, default=None,
                       help='Index of single classification endpoint to train (0-25)')
    parser.add_argument('--single_endpoint_reg', type=int, default=None,
                       help='Index of single regression endpoint to train (0-4)')

    # Head-only fine-tuning (shared trunk) options
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to a checkpoint .pth to load model weights from before training')
    parser.add_argument('--freeze_trunk', action='store_true',
                       help='Freeze shared trunk and all heads except the selected single-task head')

    args = parser.parse_args()

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("experiments") / f"{args.experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup reproducibility context (A0)
    with ReproducibilityContext(seed=args.seed, strict=args.deterministic,
                               experiment_dir=str(experiment_dir)):

        # Attach a per-experiment file handler so logs are saved under this run.
        try:
            file_handler = logging.FileHandler(experiment_dir / "train.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            # Avoid duplicate file handlers if rerun inside same process
            if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str((experiment_dir / 'train.log').resolve()) for h in logger.handlers):
                logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to attach file handler for logging: {e}")

        logger.info(f"🔬 Starting experiment: {args.experiment_name}")
        logger.info(f"📁 Experiment directory: {experiment_dir}")
        logger.info(f"🎲 Random seed: {args.seed}")
        logger.info(f"🔒 Deterministic mode: {args.deterministic}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Total GPU Memory: {gpu_memory:.2f} GB")

        # Create experiment snapshot for reproducibility
        experiment_config = vars(args)
        create_experiment_snapshot(
            str(experiment_dir),
            experiment_config,
            additional_info={
                'device': str(device),
                'gpu_memory_gb': gpu_memory if torch.cuda.is_available() else None,
                'start_time': datetime.now().isoformat()
            }
        )

        output_dir = experiment_dir / "checkpoints"
        output_dir.mkdir(exist_ok=True)
    
    config = get_enhanced_toxd4c_config()
    config['batch_size'] = args.batch_size

    # Apply ablation study modifications (A3)
    if args.disable_gnn:
        config['use_gnn'] = False
        logger.info("🧪 Ablation: GNN components disabled")
    if args.disable_transformer:
        config['use_transformer'] = False
        logger.info("🧪 Ablation: Transformer components disabled")
    if args.disable_geometric:
        config['use_geometric_encoder'] = False
        logger.info("🧪 Ablation: Geometric encoding disabled")
    if args.disable_hierarchical:
        config['use_hierarchical_encoder'] = False
        logger.info("🧪 Ablation: Hierarchical encoding disabled")
    if args.disable_fingerprint:
        config['use_fingerprints'] = False
        logger.info("🧪 Ablation: Fingerprint branch disabled")
    if args.disable_classification:
        config['enable_classification'] = False
        logger.info("🧪 Ablation: Classification tasks disabled")
    if args.disable_regression:
        config['enable_regression'] = False
        logger.info("🧪 Ablation: Regression tasks disabled")

    # Toggle contrastive learning if requested
    if getattr(args, 'disable_contrastive', False) or getattr(args, 'disable_contrastive_loss', False):
        config['use_contrastive_learning'] = False
        logger.info("🧪 Ablation: Supervised contrastive learning disabled")

    # Apply fusion method settings (for R2C11 cross-attention vs concatenation)
    if args.disable_dynamic_fusion or args.fusion_method == 'concatenation':
        config['use_dynamic_fusion'] = False
        config['fusion_method'] = 'concatenation'
        logger.info("🧪 Fusion: Using concatenation fusion instead of cross-attention")
    else:
        config['use_dynamic_fusion'] = True
        config['fusion_method'] = 'cross_attention'
        logger.info("🧪 Fusion: Using cross-attention dynamic fusion")

    # Apply GNN backbone selection
    if args.gnn_backbone == 'pyg_gcn_stack':
        config['gnn_backbone'] = 'pyg_gcn_stack'
        config['gcn_stack_layers'] = max(2, min(4, args.gcn_stack_layers))
        logger.info(f"🧪 GNN Backbone: Using PyG GCNConv stack (layers={config['gcn_stack_layers']})")
    else:
        # default
        config['gnn_backbone'] = 'graph_attention'
        logger.info("🧪 GNN Backbone: Using default GraphAttentionNetwork branch")

    # Apply task mode settings for sensitivity analysis (R1C3)
    config['task_mode'] = args.task_mode
    if args.task_mode == 'single':
        if args.single_endpoint_cls is not None:
            config['single_endpoint_cls'] = args.single_endpoint_cls
            config['enable_regression'] = False
            logger.info(f"🎯 Task Mode: Single classification endpoint {args.single_endpoint_cls}")
        elif args.single_endpoint_reg is not None:
            config['single_endpoint_reg'] = args.single_endpoint_reg
            config['enable_classification'] = False
            logger.info(f"🎯 Task Mode: Single regression endpoint {args.single_endpoint_reg}")
        else:
            logger.warning("⚠️ Single task mode specified but no endpoint index provided")
    elif args.task_mode == 'aggregated':
        # For aggregated mode, we keep all tasks but might apply different weighting
        logger.info("🎯 Task Mode: Aggregated scores (multi-task)")
    else:
        # Default multi-task mode
        logger.info("🎯 Task Mode: Multi-task (default)")

    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Handle preprocessing and data loading
    data_dir_to_use = args.data_dir

    if args.use_preprocessed:
        # Check if preprocessed data exists
        preprocessed_train = Path(args.preprocessed_dir) / "train.lmdb"
        preprocessed_valid = Path(args.preprocessed_dir) / "valid.lmdb"
        preprocessed_test = Path(args.preprocessed_dir) / "test.lmdb"

        if (preprocessed_train.exists() and preprocessed_valid.exists() and
            preprocessed_test.exists() and not args.force_preprocess):
            logger.info(f"🚀 Using preprocessed data from: {args.preprocessed_dir}")
            data_dir_to_use = args.preprocessed_dir
        else:
            logger.info("🔄 Preprocessed data not found or force_preprocess enabled")
            logger.info("⚡ Starting preprocessing to speed up training...")

            # Create preprocessed directory
            Path(args.preprocessed_dir).mkdir(parents=True, exist_ok=True)

            # Preprocess each split
            for split in ['train', 'valid', 'test']:
                input_path = Path(args.data_dir) / f"{split}.lmdb"
                output_path = Path(args.preprocessed_dir) / f"{split}.lmdb"

                if input_path.exists():
                    logger.info(f"   Preprocessing {split} split...")
                    preprocess_lmdb(str(input_path), str(output_path), args.max_atoms)
                else:
                    logger.warning(f"   Input file not found: {input_path}")

            logger.info("✅ Preprocessing completed!")
            data_dir_to_use = args.preprocessed_dir

    logger.info(f"Loading LMDB dataset from: {data_dir_to_use}")
    try:
        train_loader, valid_loader, test_loader = create_lmdb_dataloaders(
            data_dir_to_use,
            batch_size=args.batch_size,
            max_atoms=args.max_atoms
        )
        
        logger.info(f"Number of training batches: {len(train_loader)}")
        logger.info(f"Number of validation batches: {len(valid_loader)}")
        logger.info(f"Number of test batches: {len(test_loader)}")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    logger.info("Creating ToxD4C model...")
    model = ToxD4C(config, device=device).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created successfully. Total parameters: {total_params:,}")
    
    logger.info("Testing model forward pass...")
    model.eval()
    with torch.no_grad():
        try:
            sample_batch = next(iter(train_loader))
            data = {
                'atom_features': sample_batch['atom_features'].to(device),
                'edge_index': sample_batch['edge_index'].to(device),
                'coordinates': sample_batch['coordinates'].to(device),
                'batch': sample_batch['batch'].to(device)
            }
            smiles_list = sample_batch['smiles']
            
            test_outputs = model(data, smiles_list)
            
            logger.info("Test output shapes:")
            for key, value in test_outputs['predictions'].items():
                logger.info(f"  {key}: {value.shape}")
                has_nan = torch.isnan(value).any().item()
                has_inf = torch.isinf(value).any().item()
                logger.info(f"  {key} contains NaN: {has_nan}")
                logger.info(f"  {key} contains Inf: {has_inf}")
            
            if any(torch.isnan(v).any() for v in test_outputs['predictions'].values()):
                logger.error("Model output contains NaN. Please check the model implementation.")
                return
            
            logger.info("Model forward pass test successful!")
            
        except Exception as e:
            logger.error(f"Model forward pass test failed: {e}")
            logger.error(traceback.format_exc())
            return
    
    # Optionally resume from a checkpoint (e.g., full multi-task trunk)
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            try:
                checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
                missing, unexpected = model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=False)
                logger.info(f"🔁 Loaded checkpoint from: {ckpt_path}")
                if missing:
                    logger.info(f"   Missing keys: {len(missing)}")
                if unexpected:
                    logger.info(f"   Unexpected keys: {len(unexpected)}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint from {ckpt_path}: {e}")
        else:
            logger.warning(f"resume_from path does not exist: {ckpt_path}")

    # Optionally freeze trunk and all heads except the selected one (head-only FT)
    if args.freeze_trunk:
        # Determine selected head name (only valid for single-task modes)
        selected_task_name = None
        try:
            from configs.toxd4c_config import CLASSIFICATION_TASKS, REGRESSION_TASKS
            if config.get('task_mode') == 'single' and config.get('single_endpoint_cls') is not None:
                idx = int(config['single_endpoint_cls'])
                if 0 <= idx < len(CLASSIFICATION_TASKS):
                    selected_task_name = CLASSIFICATION_TASKS[idx]
            elif config.get('task_mode') == 'single' and config.get('single_endpoint_reg') is not None:
                idx = int(config['single_endpoint_reg'])
                if 0 <= idx < len(REGRESSION_TASKS):
                    selected_task_name = REGRESSION_TASKS[idx]
        except Exception:
            selected_task_name = None

        # Freeze everything by default
        for p in model.parameters():
            p.requires_grad = False

        # Unfreeze only the selected head, if present
        head_unfrozen = False
        if hasattr(model, 'prediction_head') and hasattr(model.prediction_head, 'task_heads') and selected_task_name:
            if selected_task_name in model.prediction_head.task_heads:
                for p in model.prediction_head.task_heads[selected_task_name].parameters():
                    p.requires_grad = True
                head_unfrozen = True

        if head_unfrozen:
            logger.info(f"🧩 Head-only fine-tuning: unfroze head for task '{selected_task_name}' and froze trunk.")
        else:
            logger.warning("freeze_trunk requested but could not identify the selected head. Training may have no trainable params.")

    # Build optimizer over trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        logger.warning("No trainable parameters found; enabling all parameters for optimizer as fallback.")
        trainable_params = list(model.parameters())

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=1e-5,
        eps=1e-8
    )
    
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info("Starting training...")
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_loss, train_cls_loss, train_reg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        
        val_metrics, val_loss, val_cls_loss, val_reg_loss = evaluate_model(
            model, valid_loader, device, config
        )
        
        logger.info(f"=== Epoch {epoch + 1} Results ===")
        logger.info(f"Training Loss: {train_loss:.4f} (Classification: {train_cls_loss:.4f}, Regression: {train_reg_loss:.4f})")
        logger.info(f"Validation Loss: {val_loss:.4f} (Classification: {val_cls_loss:.4f}, Regression: {val_reg_loss:.4f})")
        
        if val_metrics:
            cls_accs = [v for k, v in val_metrics.items() if 'accuracy' in k]
            cls_aucs = [v for k, v in val_metrics.items() if 'auc' in k]
            
            r2_scores = [v for k, v in val_metrics.items() if 'r2' in k]
            rmse_scores = [v for k, v in val_metrics.items() if 'rmse' in k]
            
            if cls_accs:
                avg_acc = np.mean(cls_accs)
                logger.info(f"Average Classification Accuracy: {avg_acc:.4f} (on {len(cls_accs)}/26 tasks)")
            
            if cls_aucs:
                avg_auc = np.mean(cls_aucs)
                logger.info(f"Average AUC: {avg_auc:.4f}")
            
            if r2_scores:
                avg_r2 = np.mean(r2_scores)
                logger.info(f"Average R^2: {avg_r2:.4f} (on {len(r2_scores)}/5 tasks)")
            
            if rmse_scores:
                avg_rmse = np.mean(rmse_scores)
                logger.info(f"Average RMSE: {avg_rmse:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            
            checkpoint_path = output_dir / f"{args.experiment_name}_best.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Best model saved to: {checkpoint_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Validation loss has not improved for {patience} epochs. Early stopping.")
            break
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"{args.experiment_name}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to: {checkpoint_path}")
    
    logger.info("Training finished!")
    
    logger.info("Performing final evaluation...")
    
    best_checkpoint_path = output_dir / f"{args.experiment_name}_best.pth"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from: {best_checkpoint_path}")
    
    final_metrics, final_loss, final_cls_loss, final_reg_loss = evaluate_model(
        model, valid_loader, device, config
    )
    
    logger.info(f"Final Validation Results:")
    logger.info(f"  Total Loss: {final_loss:.4f}")
    logger.info(f"  Classification Loss: {final_cls_loss:.4f}")
    logger.info(f"  Regression Loss: {final_reg_loss:.4f}")
    
    if final_metrics:
        cls_accs = [v for k, v in final_metrics.items() if 'accuracy' in k]
        cls_aucs = [v for k, v in final_metrics.items() if 'auc' in k]
        
        if cls_accs:
            logger.info(f"  Average Classification Accuracy: {np.mean(cls_accs):.4f}")
        if cls_aucs:
            logger.info(f"  Average AUC: {np.mean(cls_aucs):.4f}")
        
        r2_scores = [v for k, v in final_metrics.items() if 'r2' in k]
        rmse_scores = [v for k, v in final_metrics.items() if 'rmse' in k]
        
        if r2_scores:
            logger.info(f"  Average R^2: {np.mean(r2_scores):.4f}")
        if rmse_scores:
            logger.info(f"  Average RMSE: {np.mean(rmse_scores):.4f}")
    
    results = {
        'experiment_name': args.experiment_name,
        'config': config,
        'final_metrics': final_metrics,
        'final_loss': final_loss,
        'final_cls_loss': final_cls_loss,
        'final_reg_loss': final_reg_loss,
        'model_params': total_params
    }
    
    results_path = output_dir / f"{args.experiment_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
