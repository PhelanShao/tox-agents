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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error, r2_score,
    average_precision_score, matthews_corrcoef, precision_recall_curve
)

sys.path.insert(0, str(Path(__file__).parent))

from data.lmdb_dataset import create_lmdb_dataloaders
from models.toxd4c import ToxD4C
from configs.toxd4c_config import get_experiment_config
from models.losses.focal_loss import FocalLoss, compute_class_weights
from training.splits import build_scaffold_lmdb_splits, build_scaffold_lmdb_splits_from_dir
from preprocess_data import preprocess_lmdb

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_real_data.log')
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


def compute_metrics(predictions, targets, masks, task_type='classification', task_names=None):
    """
    Compute evaluation metrics with imbalance-robust measures.
    
    For classification:
        - Accuracy
        - ROC-AUC
        - PR-AUC (Precision-Recall AUC) - robust to class imbalance
        - MCC (Matthews Correlation Coefficient) - balanced measure
    
    For regression:
        - MSE, RMSE, R²
    """
    metrics = {}
    all_aucs = []
    all_pr_aucs = []
    all_mccs = []
    
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
        
        task_name = task_names[task_idx] if task_names and task_idx < len(task_names) else f'task_{task_idx}'
        
        try:
            if task_type == 'classification':
                pred_binary = (valid_pred > 0.5).astype(int)
                target_binary = valid_target.astype(int)
                
                # Accuracy
                acc = accuracy_score(target_binary, pred_binary)
                metrics[f'{task_name}_accuracy'] = acc
                
                if len(np.unique(target_binary)) > 1:
                    # ROC-AUC
                    auc = roc_auc_score(target_binary, valid_pred)
                    metrics[f'{task_name}_auc'] = auc
                    all_aucs.append(auc)
                    
                    # PR-AUC (more robust to class imbalance)
                    pr_auc = average_precision_score(target_binary, valid_pred)
                    metrics[f'{task_name}_pr_auc'] = pr_auc
                    all_pr_aucs.append(pr_auc)
                    
                    # MCC (Matthews Correlation Coefficient)
                    mcc = matthews_corrcoef(target_binary, pred_binary)
                    metrics[f'{task_name}_mcc'] = mcc
                    all_mccs.append(mcc)
            else:
                mse = mean_squared_error(valid_target, valid_pred)
                if np.var(valid_target) < 1e-6:
                    r2 = 0.0
                else:
                    r2 = r2_score(valid_target, valid_pred)
                rmse = np.sqrt(mse)
                
                metrics[f'{task_name}_mse'] = mse
                metrics[f'{task_name}_rmse'] = rmse
                metrics[f'{task_name}_r2'] = r2
        except Exception as e:
            logger.warning(f"Error computing metrics for {task_name}: {e}")
            continue
    
    # Add aggregate metrics for classification
    if task_type == 'classification':
        if all_aucs:
            metrics['mean_auc'] = np.mean(all_aucs)
        if all_pr_aucs:
            metrics['mean_pr_auc'] = np.mean(all_pr_aucs)
        if all_mccs:
            metrics['mean_mcc'] = np.mean(all_mccs)
    
    return metrics


def train_epoch(model, dataloader, optimizer, device, pos_weights=None):
    """
    Train for one epoch with Focal Loss and class weighting.
    
    Args:
        model: ToxD4C model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device (cpu/cuda)
        pos_weights: Optional pre-computed positive class weights [num_cls_tasks]
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    # Use Focal Loss with gamma=2.0 for classification (handles class imbalance)
    # If pos_weights provided, use inverse frequency weighting
    focal_gamma = model.config.get('focal_gamma', 2.0)
    if pos_weights is not None:
        classification_criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weights.to(device), reduction='none')
    else:
        classification_criterion = FocalLoss(gamma=focal_gamma, reduction='none')
    
    regression_criterion = nn.MSELoss(reduction='none')

    # Get task weights from config
    task_weights = model.config.get('task_weights', {})
    cls_task_names = model.config.get('classification_tasks_list', [])
    reg_task_names = model.config.get('regression_tasks_list', [])
    
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
            smiles_list = batch['smiles']
            
            optimizer.zero_grad()
            
            outputs = model(data, smiles_list)
            
            cls_preds = outputs['predictions']['classification']
            reg_preds = outputs['predictions']['regression']

            if check_for_nan_inf(cls_preds, "classification_output"):
                logger.warning(f"NaN in classification output at batch {batch_idx}")
                continue
            
            if check_for_nan_inf(reg_preds, "regression_output"):
                logger.warning(f"NaN in regression output at batch {batch_idx}")
                continue
            
            # Focal Loss with mask-aware computation
            cls_loss = classification_criterion(cls_preds, cls_labels, mask=cls_mask)
            
            reg_loss = safe_loss_computation(
                reg_preds, reg_labels, reg_mask,
                lambda p, t: regression_criterion(p, t).mean()
            )
            
            # Apply task weights with Focal Loss
            weighted_cls_loss = 0.0
            if cls_preds.numel() > 0 and len(cls_task_names) > 0:
                for i, task_name in enumerate(cls_task_names):
                    task_weight = task_weights.get(task_name, 1.0)
                    task_mask = cls_mask[:, i]
                    if task_mask.sum() > 0:
                        # Per-task focal loss
                        task_loss = classification_criterion(
                            cls_preds[:, i:i+1], 
                            cls_labels[:, i:i+1], 
                            mask=task_mask.unsqueeze(1)
                        )
                        weighted_cls_loss += task_loss * task_weight
                weighted_cls_loss /= len(cls_task_names)
            else:
                weighted_cls_loss = cls_loss  # Fallback if no classification tasks

            weighted_reg_loss = 0.0
            if reg_preds.numel() > 0:
                for i, task_name in enumerate(reg_task_names):
                    weight = task_weights.get(task_name, 1.0)
                    weighted_reg_loss += safe_loss_computation(
                        reg_preds[:, i], reg_labels[:, i], reg_mask[:, i],
                        lambda p, t: regression_criterion(p, t).mean()
                    ) * weight
                weighted_reg_loss /= len(reg_task_names) if len(reg_task_names) > 0 else 1.0
            else:
                weighted_reg_loss = reg_loss # Fallback if no regression tasks

            total_loss_batch = weighted_cls_loss + weighted_reg_loss
            
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
            
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}: "
                          f"Loss={total_loss_batch.item():.4f}, "
                          f"Cls={cls_loss.item():.4f}, "
                          f"Reg={reg_loss.item():.4f}")
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    
    return total_loss / num_batches, total_cls_loss / num_batches, total_reg_loss / num_batches


def evaluate_model(model, dataloader, device):
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
    
    # Get task weights from config
    task_weights = model.config.get('task_weights', {})
    cls_task_names = model.config.get('classification_tasks_list', [])
    reg_task_names = model.config.get('regression_tasks_list', [])
    
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
                smiles_list = batch['smiles']
                
                outputs = model(data, smiles_list)
                
                cls_preds = outputs['predictions']['classification']
                reg_preds = outputs['predictions']['regression']

                if check_for_nan_inf(cls_preds, "eval_classification"):
                    continue
                if check_for_nan_inf(reg_preds, "eval_regression"):
                    continue
                
                # Calculate unweighted losses for logging
                unweighted_cls_loss = safe_loss_computation(
                    cls_preds, cls_labels, cls_mask,
                    lambda p, t: classification_criterion(p, t).mean()
                )
                
                unweighted_reg_loss = safe_loss_computation(
                    reg_preds, reg_labels, reg_mask,
                    lambda p, t: regression_criterion(p, t).mean()
                )

                # Apply task weights for the main validation loss
                weighted_cls_loss = 0.0
                if cls_preds.numel() > 0:
                    for i, task_name in enumerate(cls_task_names):
                        weight = task_weights.get(task_name, 1.0)
                        weighted_cls_loss += safe_loss_computation(
                            cls_preds[:, i], cls_labels[:, i], cls_mask[:, i],
                            lambda p, t: classification_criterion(p, t).mean()
                        ) * weight
                    weighted_cls_loss /= len(cls_task_names) if len(cls_task_names) > 0 else 1.0
                else:
                    weighted_cls_loss = unweighted_cls_loss

                weighted_reg_loss = 0.0
                if reg_preds.numel() > 0:
                    for i, task_name in enumerate(reg_task_names):
                        weight = task_weights.get(task_name, 1.0)
                        weighted_reg_loss += safe_loss_computation(
                            reg_preds[:, i], reg_labels[:, i], reg_mask[:, i],
                            lambda p, t: regression_criterion(p, t).mean()
                        ) * weight
                    weighted_reg_loss /= len(reg_task_names) if len(reg_task_names) > 0 else 1.0
                else:
                    weighted_reg_loss = unweighted_reg_loss
                
                total_loss_batch = weighted_cls_loss + weighted_reg_loss
                
                total_loss += total_loss_batch.item()
                total_cls_loss += unweighted_cls_loss.item()
                total_reg_loss += unweighted_reg_loss.item()
                num_batches += 1
                
                cls_probs = torch.sigmoid(cls_preds)
                all_cls_preds.append(cls_probs.cpu())
                all_cls_targets.append(cls_labels.cpu())
                all_cls_masks.append(cls_mask.cpu())
                
                all_reg_preds.append(reg_preds.cpu())
                all_reg_targets.append(reg_labels.cpu())
                all_reg_masks.append(reg_mask.cpu())
                
            except Exception as e:
                logger.error(f"Error in evaluation batch: {e}")
                continue
    
    if num_batches == 0:
        return {}, 0.0, 0.0, 0.0
    
    metrics = {}
    
    cls_task_names = model.config.get('classification_tasks_list', [])
    reg_task_names = model.config.get('regression_tasks_list', [])
    
    if all_cls_preds:
        cls_preds = torch.cat(all_cls_preds, dim=0)
        cls_targets = torch.cat(all_cls_targets, dim=0)
        cls_masks = torch.cat(all_cls_masks, dim=0)
        
        cls_metrics = compute_metrics(cls_preds, cls_targets, cls_masks, 'classification', task_names=cls_task_names)
        metrics.update(cls_metrics)
    
    if all_reg_preds:
        reg_preds = torch.cat(all_reg_preds, dim=0)
        reg_targets = torch.cat(all_reg_targets, dim=0)
        reg_masks = torch.cat(all_reg_masks, dim=0)
        
        reg_metrics = compute_metrics(reg_preds, reg_targets, reg_masks, 'regression', task_names=reg_task_names)
        metrics.update(reg_metrics)
    
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    
    return metrics, avg_loss, avg_cls_loss, avg_reg_loss


def main():
    parser = argparse.ArgumentParser(description='ToxD4C Training Script with optional scaffold split')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Directory for PREPROCESSED LMDB data')
    parser.add_argument('--experiment_name', type=str, default='full_model',
                        help='Name of the experiment config to run (e.g., full_model, gnn_only, etc.)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--max_atoms', type=int, default=64, help='Maximum number of atoms')
    parser.add_argument('--warmup_ratio', type=float, default=0.06, help='Learning rate warmup ratio')
    parser.add_argument('--split_method', type=str, default='random', choices=['random', 'scaffold'],
                        help='Dataset split method. "random" uses the provided splits; "scaffold" rebuilds splits by Bemis–Murcko scaffold.')
    parser.add_argument('--save_splits', action='store_true', help='Save split indices JSON when building scaffold splits')
    parser.add_argument('--limit_per_split', type=int, default=None, help='Limit number of samples per split for quick runs')
    parser.add_argument('--scaffold_copy_limit', type=int, default=None, help='When building scaffold splits from processed LMDB, limit total molecules to copy for a quick run')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Total GPU Memory: {gpu_memory:.2f} GB")
    
    # Create a subdirectory for the specific experiment
    output_dir = Path("checkpoints_real") / args.experiment_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get the configuration for the specified experiment
    try:
        config = get_experiment_config(args.experiment_name)
    except ValueError as e:
        logger.error(e)
        return
    config['batch_size'] = args.batch_size
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")
    
    # If requested, build scaffold-split LMDBs and pre-process them
    if args.split_method == 'scaffold':
        processed_dir = Path('data/processed')
        processed_scaffold = Path('data/processed_scaffold')
        args.data_dir = str(processed_scaffold)

        if not (processed_scaffold / 'train.lmdb').exists():
            splits_path = Path('splits') / 'scaffold_indices.json' if args.save_splits else None
            if (processed_dir / 'train.lmdb').exists():
                logger.info('Building scaffold-based LMDB splits by copying processed entries (fast path)...')
                _ = build_scaffold_lmdb_splits_from_dir(
                    input_dir=processed_dir,
                    out_dir=processed_scaffold,
                    seed=42,
                    frac=(0.8, 0.1, 0.1),
                    save_splits_path=str(splits_path) if splits_path else None,
                    limit_total=args.scaffold_copy_limit,
                )
            else:
                logger.info('Processed LMDB not found; falling back to raw split + preprocessing (slow path)...')
                raw_dir = Path('data/dataset')
                scaffold_raw = Path('data/dataset_scaffold')
                _ = build_scaffold_lmdb_splits(
                    raw_dir=raw_dir,
                    out_dir=scaffold_raw,
                    seed=42,
                    frac=(0.8, 0.1, 0.1),
                    save_splits_path=str(splits_path) if splits_path else None,
                )
                processed_scaffold.mkdir(parents=True, exist_ok=True)
                for split in ['train', 'valid', 'test']:
                    input_path = str(scaffold_raw / f'{split}.lmdb')
                    output_path = str(processed_scaffold / f'{split}.lmdb')
                    preprocess_lmdb(input_path, output_path, max_atoms=args.max_atoms)

    logger.info(f"Loading LMDB dataset from: {args.data_dir}")
    try:
        train_loader, valid_loader, test_loader = create_lmdb_dataloaders(
            args.data_dir, 
            batch_size=args.batch_size,
            max_atoms=args.max_atoms,
            limit_per_split=args.limit_per_split
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
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
        eps=1e-8
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',      # The scheduler will reduce LR when the metric stops decreasing
        factor=0.3,      # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5      # Number of epochs with no improvement after which learning rate will be reduced.
    )
    
    logger.info("Starting training with ReduceLROnPlateau scheduler...")
    
    # Compute class weights for handling class imbalance (inverse frequency weighting)
    logger.info("Computing class weights for imbalanced data...")
    pos_weights = None
    try:
        all_labels = []
        all_masks = []
        for batch in train_loader:
            all_labels.append(batch['classification_labels'])
            all_masks.append(batch['classification_mask'])
        
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        pos_weights = compute_class_weights(
            all_labels, all_masks, 
            method='inverse_freq',  # Use inverse frequency weighting
            clip_range=(0.1, 10.0)  # Prevent extreme weights
        )
        logger.info(f"Computed pos_weights (sample): {pos_weights[:5].tolist()}")
        logger.info(f"Class weight range: [{pos_weights.min():.2f}, {pos_weights.max():.2f}]")
    except Exception as e:
        logger.warning(f"Could not compute class weights: {e}. Using uniform weights.")
        pos_weights = None
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_loss, train_cls_loss, train_reg_loss = train_epoch(
            model, train_loader, optimizer, device, pos_weights=pos_weights
        )
        
        val_metrics, val_loss, val_cls_loss, val_reg_loss = evaluate_model(
            model, valid_loader, device
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
                logger.info(f"Average ROC-AUC: {avg_auc:.4f}")
            
            # Log imbalance-robust metrics
            pr_aucs = [v for k, v in val_metrics.items() if 'pr_auc' in k]
            mccs = [v for k, v in val_metrics.items() if 'mcc' in k]
            
            if pr_aucs:
                avg_pr_auc = np.mean(pr_aucs)
                logger.info(f"Average PR-AUC: {avg_pr_auc:.4f}")
            
            if mccs:
                avg_mcc = np.mean(mccs)
                logger.info(f"Average MCC: {avg_mcc:.4f}")
            
            if r2_scores:
                avg_r2 = np.mean(r2_scores)
                logger.info(f"Average R²: {avg_r2:.4f} (on {len(r2_scores)}/5 tasks)")
            
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
                'config': config,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            
            checkpoint_path = output_dir / f"{args.experiment_name}_best.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Best model saved to: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Step the scheduler on the validation loss
        scheduler.step(val_loss)
        
        if patience_counter >= patience:
            logger.info(f"Validation loss has not improved for {patience} epochs. Early stopping.")
            break
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"{args.experiment_name}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
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
        model, valid_loader, device
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
            logger.info(f"  Average R²: {np.mean(r2_scores):.4f}")
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

    # Evaluate on test set and export a minimal metrics JSON for R1C1 figures
    test_metrics, test_loss, test_cls_loss, test_reg_loss = evaluate_model(
        model, test_loader, device
    )
    results_dir = Path('results/generalization')
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.split_method == 'scaffold':
        out_json = results_dir / 'scaffold_metrics.json'
    else:
        out_json = results_dir / 'random_metrics.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
