"""
Train ToxD4C with multi-conformer data augmentation.
Uses all 11 conformers from original dataset for training.
With comprehensive metrics logging for each epoch.
"""

import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error

sys.path.insert(0, str(Path(__file__).parent))


class MetricsTracker:
    """Track and save training metrics for each epoch."""

    def __init__(self, output_dir: str, experiment_name: str = "multi_conformer"):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.metrics_history = []
        self.csv_path = os.path.join(output_dir, f'{experiment_name}_metrics.csv')
        self.json_path = os.path.join(output_dir, f'{experiment_name}_metrics.json')

        # Initialize CSV file with headers
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            'epoch', 'timestamp',
            'train_loss', 'train_cls_loss', 'train_reg_loss',
            'val_roc_auc', 'val_mse', 'val_cls_tasks', 'val_reg_tasks',
            'learning_rate', 'epoch_time_sec'
        ]
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics for one epoch."""
        metrics['epoch'] = epoch
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)

        # Append to CSV
        row = [
            metrics.get('epoch', 0),
            metrics.get('timestamp', ''),
            metrics.get('train_loss', 0),
            metrics.get('train_cls_loss', 0),
            metrics.get('train_reg_loss', 0),
            metrics.get('val_roc_auc', 0),
            metrics.get('val_mse', 0),
            metrics.get('val_cls_tasks', 0),
            metrics.get('val_reg_tasks', 0),
            metrics.get('learning_rate', 0),
            metrics.get('epoch_time_sec', 0)
        ]
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Save full JSON after each epoch (for safety)
        self._save_json()

    def _save_json(self):
        """Save all metrics to JSON file."""
        with open(self.json_path, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'total_epochs': len(self.metrics_history),
                'metrics_history': self.metrics_history
            }, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics_history:
            return {}

        val_aucs = [m.get('val_roc_auc', 0) for m in self.metrics_history]
        train_losses = [m.get('train_loss', 0) for m in self.metrics_history]

        best_epoch_idx = np.argmax(val_aucs)

        return {
            'best_epoch': self.metrics_history[best_epoch_idx]['epoch'],
            'best_val_auc': max(val_aucs),
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'total_epochs': len(self.metrics_history)
        }

from data.multi_conformer_dataset import (
    OriginalMultiConformerDataset, 
    collate_multi_conformer_batch,
    create_original_multi_conformer_dataloaders
)
from models.toxd4c import ToxD4C
from configs.toxd4c_config import get_experiment_config

warnings.filterwarnings('ignore', category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_multi_conformer.log')
    ]
)
logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, optimizer, device, config, grad_clip=1.0, grad_clip_value=None):
    """Train for one epoch. Returns detailed loss breakdown.

    Args:
        grad_clip: Max norm for gradient clipping (default: 1.0)
        grad_clip_value: Optional value-based gradient clipping
    """
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    nan_batches = 0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        try:
            # Move data to device
            data = {
                'atom_features': batch['atom_features'].to(device),
                'edge_index': batch['edge_index'].to(device),
                'coordinates': batch['coordinates'].to(device),
                'batch': batch['batch'].to(device)
            }

            smiles_list = batch['smiles']  # List of SMILES strings

            cls_labels = batch['classification_labels'].to(device)
            reg_labels = batch['regression_labels'].to(device)
            cls_mask = batch['classification_mask'].to(device)
            reg_mask = batch['regression_mask'].to(device)

            optimizer.zero_grad()

            # Forward pass - model needs data dict and smiles_list
            outputs = model(data, smiles_list)
            predictions = outputs['predictions']
            cls_logits = predictions['classification']  # shape: (batch, num_cls_tasks)
            reg_preds = predictions['regression']  # shape: (batch, num_reg_tasks)

            # Classification loss (BCE with mask)
            cls_loss = torch.tensor(0.0, device=device)
            if not config.get('disable_classification', False):
                cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                cls_loss = cls_loss_fn(cls_logits, cls_labels)
                cls_loss = (cls_loss * cls_mask.float()).sum() / (cls_mask.sum() + 1e-8)

            # Regression loss (MSE with mask)
            reg_loss = torch.tensor(0.0, device=device)
            if not config.get('disable_regression', False):
                reg_loss_fn = nn.MSELoss(reduction='none')
                reg_loss = reg_loss_fn(reg_preds, reg_labels)
                reg_loss = (reg_loss * reg_mask.float()).sum() / (reg_mask.sum() + 1e-8)

            # Combined loss
            loss = cls_loss + config.get('regression_weight', 0.1) * reg_loss

            # Check for NaN loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                if nan_batches <= 5:  # Only log first 5 NaN batches per epoch
                    logger.warning(f"  Batch {batch_idx}: NaN/Inf loss detected, skipping...")
                elif nan_batches == 6:
                    logger.warning(f"  Suppressing further NaN warnings for this epoch...")
                optimizer.zero_grad()
                continue

            loss.backward()

            # Check for NaN gradients before clipping
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                nan_batches += 1
                if nan_batches <= 5:
                    logger.warning(f"  Batch {batch_idx}: NaN/Inf gradient detected, skipping...")
                optimizer.zero_grad()
                continue

            # Apply gradient clipping
            if grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=grad_clip_value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1

            if batch_idx % 200 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)}: loss={loss.item():.4f}, cls={cls_loss.item():.4f}, reg={reg_loss.item():.4f}")

        except Exception as e:
            logger.warning(f"Error in batch {batch_idx}: {e}")
            continue

    n = max(num_batches, 1)

    # Log NaN statistics for this epoch
    if nan_batches > 0:
        logger.warning(f"  Epoch had {nan_batches} NaN/Inf batches out of {len(train_loader)} total batches")

    return {
        'total_loss': total_loss / n,
        'cls_loss': total_cls_loss / n,
        'reg_loss': total_reg_loss / n,
        'nan_batches': nan_batches,
        'valid_batches': num_batches
    }


def evaluate(model, data_loader, device, config=None):
    """Evaluate model on validation/test set."""
    model.eval()

    if config is None:
        config = {}

    all_cls_preds = []
    all_cls_labels = []
    all_cls_masks = []
    all_reg_preds = []
    all_reg_labels = []
    all_reg_masks = []

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue

            try:
                data = {
                    'atom_features': batch['atom_features'].to(device),
                    'edge_index': batch['edge_index'].to(device),
                    'coordinates': batch['coordinates'].to(device),
                    'batch': batch['batch'].to(device)
                }

                smiles_list = batch['smiles']

                outputs = model(data, smiles_list)
                predictions = outputs['predictions']
                cls_probs = torch.sigmoid(predictions['classification'])
                reg_preds = predictions['regression']

                all_cls_preds.append(cls_probs.cpu().numpy())
                all_cls_labels.append(batch['classification_labels'].numpy())
                all_cls_masks.append(batch['classification_mask'].numpy())
                all_reg_preds.append(reg_preds.cpu().numpy())
                all_reg_labels.append(batch['regression_labels'].numpy())
                all_reg_masks.append(batch['regression_mask'].numpy())

            except Exception as e:
                logger.warning(f"Error in evaluation: {e}")
                continue

    if not all_cls_preds:
        return {'roc_auc': 0.0, 'mse': float('inf')}

    cls_preds = np.concatenate(all_cls_preds, axis=0)
    cls_labels = np.concatenate(all_cls_labels, axis=0)
    cls_masks = np.concatenate(all_cls_masks, axis=0)
    reg_preds = np.concatenate(all_reg_preds, axis=0)
    reg_labels = np.concatenate(all_reg_labels, axis=0)
    reg_masks = np.concatenate(all_reg_masks, axis=0)

    # Calculate per-task ROC-AUC (with NaN handling)
    roc_aucs = []
    if not config.get('disable_classification', False):
        for i in range(cls_labels.shape[1]):
            mask = cls_masks[:, i].astype(bool)
            if mask.sum() > 10:
                labels_i = cls_labels[mask, i]
                preds_i = cls_preds[mask, i]
                # Filter out NaN values
                valid_mask = ~(np.isnan(labels_i) | np.isnan(preds_i))
                if valid_mask.sum() > 10 and len(np.unique(labels_i[valid_mask])) > 1:
                    try:
                        auc = roc_auc_score(labels_i[valid_mask], preds_i[valid_mask])
                        roc_aucs.append(auc)
                    except:
                        pass

    # Calculate MSE for regression (with NaN handling)
    mses = []
    if not config.get('disable_regression', False):
        for i in range(reg_labels.shape[1]):
            mask = reg_masks[:, i].astype(bool)
            if mask.sum() > 0:
                labels_i = reg_labels[mask, i]
                preds_i = reg_preds[mask, i]
                # Filter out NaN values
                valid_mask = ~(np.isnan(labels_i) | np.isnan(preds_i))
                if valid_mask.sum() > 0:
                    try:
                        mse = mean_squared_error(labels_i[valid_mask], preds_i[valid_mask])
                        mses.append(mse)
                    except:
                        pass

    return {
        'roc_auc': np.mean(roc_aucs) if roc_aucs else 0.0,
        'mse': np.mean(mses) if mses else float('inf'),
        'num_cls_tasks': len(roc_aucs),
        'num_reg_tasks': len(mses)
    }


def main():
    parser = argparse.ArgumentParser(description='Train ToxD4C with multi-conformer augmentation')
    parser.add_argument('--data_dir', type=str, default='data/dataset',
                        help='Directory containing original LMDB files')
    parser.add_argument('--conformer_mode', type=str, default='all',
                        choices=['all', 'random', 'random_n', 'first'],
                        help='Conformer selection mode for training')
    parser.add_argument('--n_conformers', type=int, default=11,
                        help='Number of conformers in original data')
    parser.add_argument('--sample_n_conformers', type=int, default=None,
                        help='Number of conformers to randomly sample (for random_n mode, e.g., 3)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs/multi_conformer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--experiment_name', type=str, default='multi_conformer_100epochs',
                        help='Name for this experiment (used in metrics files)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0)')
    parser.add_argument('--grad_clip_value', type=float, default=None,
                        help='Gradient clipping by value (optional, e.g., 0.5)')

    # Data split arguments
    parser.add_argument('--split_method', type=str, default='original',
                        choices=['original', 'scaffold'],
                        help='Data split method: original (use existing splits) or scaffold (Bemis-Murcko scaffold split)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio for scaffold split (default: 0.8)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Validation set ratio for scaffold split (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for scaffold split')

    # Ablation study arguments
    parser.add_argument('--disable_transformer', action='store_true',
                        help='Disable Transformer, use GNN only')
    parser.add_argument('--disable_geometric', action='store_true',
                        help='Disable 3D geometric encoder')
    parser.add_argument('--disable_hierarchical', action='store_true',
                        help='Disable hierarchical encoder')
    parser.add_argument('--disable_fingerprint', action='store_true',
                        help='Disable molecular fingerprints')
    parser.add_argument('--disable_contrastive', action='store_true',
                        help='Disable contrastive learning loss')
    parser.add_argument('--disable_dynamic_fusion', action='store_true',
                        help='Use concatenation fusion instead of dynamic fusion')
    parser.add_argument('--disable_regression', action='store_true',
                        help='Disable regression tasks (classification only)')
    parser.add_argument('--disable_classification', action='store_true',
                        help='Disable classification tasks (regression only)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ToxD4C Multi-Conformer Training (100 Epochs)")
    logger.info("=" * 60)
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Split method: {args.split_method}")
    logger.info(f"Conformer mode: {args.conformer_mode}")
    logger.info(f"Number of conformers: {args.n_conformers}")
    if args.conformer_mode == 'random_n' and args.sample_n_conformers:
        logger.info(f"Sampling {args.sample_n_conformers} conformers per molecule")
    if args.split_method == 'scaffold':
        logger.info(f"Scaffold split ratios: train={args.train_ratio}, valid={args.valid_ratio}, test={1-args.train_ratio-args.valid_ratio:.2f}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Early stopping patience: {args.patience}")
    logger.info(f"Device: {args.device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(args.output_dir, args.experiment_name)
    logger.info(f"Metrics will be saved to: {metrics_tracker.csv_path}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, valid_loader, test_loader = create_original_multi_conformer_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conformer_mode=args.conformer_mode,
        n_conformers=args.n_conformers,
        sample_n_conformers=args.sample_n_conformers,
        split_method=args.split_method,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Valid samples: {len(valid_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Get config and create model
    config = get_experiment_config('full_model')

    # Apply ablation settings
    ablation_settings = []
    if args.disable_transformer:
        config['use_hybrid_architecture'] = False
        ablation_settings.append('no_transformer')
    if args.disable_geometric:
        config['use_geometric_encoder'] = False
        ablation_settings.append('no_3d')
    if args.disable_hierarchical:
        config['use_hierarchical_encoder'] = False
        ablation_settings.append('no_hierarchical')
    if args.disable_fingerprint:
        config['use_fingerprints'] = False
        ablation_settings.append('no_fingerprint')
    if args.disable_contrastive:
        config['use_contrastive_learning'] = False
        ablation_settings.append('no_contrastive')
    if args.disable_dynamic_fusion:
        config['fusion_method'] = 'concatenation'
        ablation_settings.append('concat_fusion')
    if args.disable_regression:
        config['disable_regression'] = True
        ablation_settings.append('cls_only')
    if args.disable_classification:
        config['disable_classification'] = True
        ablation_settings.append('reg_only')

    if ablation_settings:
        logger.info(f"Ablation settings: {', '.join(ablation_settings)}")

    logger.info(f"Config - use_hybrid_architecture: {config.get('use_hybrid_architecture', False)}")
    logger.info(f"Config - use_geometric_encoder: {config.get('use_geometric_encoder', False)}")
    logger.info(f"Config - use_fingerprints: {config.get('use_fingerprints', False)}")
    logger.info(f"Config - use_hierarchical_encoder: {config.get('use_hierarchical_encoder', False)}")
    logger.info(f"Config - use_contrastive_learning: {config.get('use_contrastive_learning', False)}")

    model = ToxD4C(config=config, device=args.device)
    model = model.to(args.device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    # Training loop
    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    logger.info("\n" + "=" * 60)
    logger.info("Starting Training...")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch}/{args.epochs} {'='*20}")

        # Train
        start_time = time.time()
        train_losses = train_epoch(
            model, train_loader, optimizer, args.device, config,
            grad_clip=args.grad_clip,
            grad_clip_value=args.grad_clip_value
        )
        train_time = time.time() - start_time

        # Evaluate
        val_metrics = evaluate(model, valid_loader, args.device, config)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics for this epoch
        epoch_metrics = {
            'train_loss': train_losses['total_loss'],
            'train_cls_loss': train_losses['cls_loss'],
            'train_reg_loss': train_losses['reg_loss'],
            'val_roc_auc': val_metrics['roc_auc'],
            'val_mse': val_metrics['mse'],
            'val_cls_tasks': val_metrics.get('num_cls_tasks', 26),
            'val_reg_tasks': val_metrics.get('num_reg_tasks', 5),
            'learning_rate': current_lr,
            'epoch_time_sec': train_time
        }
        metrics_tracker.log_epoch(epoch, epoch_metrics)

        # Print summary
        logger.info(f"Train Loss: {train_losses['total_loss']:.4f} (cls: {train_losses['cls_loss']:.4f}, reg: {train_losses['reg_loss']:.4f})")
        logger.info(f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | Val MSE: {val_metrics['mse']:.4f}")
        logger.info(f"Learning Rate: {current_lr:.2e} | Time: {train_time:.1f}s")

        # Update scheduler
        scheduler.step(val_metrics['roc_auc'])

        # Save best model
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'val_mse': val_metrics['mse'],
                'config': config
            }, os.path.join(args.output_dir, 'best_model.pt'))
            logger.info(f"  ★ New best model saved! (ROC-AUC: {best_val_auc:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epochs (best: {best_val_auc:.4f} at epoch {best_epoch})")

        # Early stopping check
        if epochs_without_improvement >= args.patience:
            logger.info(f"\n⚠ Early stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs)")
            break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_metrics['roc_auc'],
                'config': config
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
            logger.info(f"  Checkpoint saved at epoch {epoch}")

    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 60)

    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, args.device, config)
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test MSE: {test_metrics['mse']:.4f}")
    logger.info(f"Best epoch: {best_epoch}")

    # Save final results
    final_results = {
        'experiment_name': args.experiment_name,
        'conformer_mode': args.conformer_mode,
        'n_conformers': args.n_conformers,
        'split_method': args.split_method,
        'train_ratio': args.train_ratio if args.split_method == 'scaffold' else None,
        'valid_ratio': args.valid_ratio if args.split_method == 'scaffold' else None,
        'total_epochs_run': epoch,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_roc_auc': test_metrics['roc_auc'],
        'test_mse': test_metrics['mse'],
        'test_cls_tasks': test_metrics.get('num_cls_tasks', 26),
        'test_reg_tasks': test_metrics.get('num_reg_tasks', 5),
        'metrics_csv': metrics_tracker.csv_path,
        'metrics_json': metrics_tracker.json_path
    }
    with open(os.path.join(args.output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    summary = metrics_tracker.get_summary()
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Total epochs run: {epoch}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation ROC-AUC: {best_val_auc:.4f}")
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"Test MSE: {test_metrics['mse']:.4f}")
    logger.info(f"\nMetrics saved to:")
    logger.info(f"  CSV: {metrics_tracker.csv_path}")
    logger.info(f"  JSON: {metrics_tracker.json_path}")
    logger.info("\nTraining complete!")


if __name__ == '__main__':
    main()

