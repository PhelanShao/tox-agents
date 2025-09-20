#!/usr/bin/env python3
"""
Evaluate a trained ToxD4C checkpoint on the test set and report metrics.

Usage example:

python ToxD4C/evaluate_test.py \
  --experiment_dir ToxD4C/experiments/toxd4c_ablation_gnn_trans_3d_20250906_142035 \
  --data_dir data/data/processed \
  --batch_size 16

This loads the best checkpoint under <experiment_dir>/checkpoints, constructs
the model using the saved config.json, evaluates on test.lmdb, and writes
test_results.json back to the same checkpoints directory.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from data.lmdb_dataset import create_lmdb_dataloaders
from models.toxd4c import ToxD4C


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _safe_loss(pred, target, mask, loss_fn, device):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    return loss_fn(pred[mask], target[mask])


@torch.no_grad()
def evaluate_on_loader(model: torch.nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: torch.device) -> Tuple[Dict[str, float], float, float, float]:
    model.eval()

    bce = nn.BCEWithLogitsLoss(reduction='none')
    mse = nn.MSELoss(reduction='none')

    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    n_batches = 0

    cls_preds_all = []
    cls_targets_all = []
    cls_masks_all = []
    reg_preds_all = []
    reg_targets_all = []
    reg_masks_all = []

    for batch in dataloader:
        if batch is None:
            # Skip empty batch produced by collate if all samples invalid
            continue

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

        outputs = model(data, batch['smiles'])
        cls_logits = outputs['predictions']['classification']
        reg_preds = outputs['predictions']['regression']

        # Determine which branches are active
        has_cls = cls_logits.numel() > 0
        has_reg = reg_preds.numel() > 0

        if has_cls:
            batch_cls_loss = _safe_loss(cls_logits, cls_labels, cls_mask, lambda p, t: bce(p, t).mean(), device)
        else:
            batch_cls_loss = torch.tensor(0.0, device=device)

        if has_reg:
            batch_reg_loss = _safe_loss(reg_preds, reg_labels, reg_mask, lambda p, t: mse(p, t).mean(), device)
        else:
            batch_reg_loss = torch.tensor(0.0, device=device)

        batch_total = batch_cls_loss + batch_reg_loss

        total_loss += float(batch_total)
        total_cls_loss += float(batch_cls_loss)
        total_reg_loss += float(batch_reg_loss)
        n_batches += 1

        if has_cls:
            cls_probs = torch.sigmoid(cls_logits)
            cls_preds_all.append(cls_probs.cpu())
            cls_targets_all.append(cls_labels.cpu())
            cls_masks_all.append(cls_mask.cpu())

        if has_reg:
            reg_preds_all.append(reg_preds.cpu())
            reg_targets_all.append(reg_labels.cpu())
            reg_masks_all.append(reg_mask.cpu())

    if n_batches == 0:
        return {}, 0.0, 0.0, 0.0

    # Aggregate
    metrics: Dict[str, float] = {}
    avg_loss = total_loss / n_batches
    avg_cls_loss = total_cls_loss / n_batches
    avg_reg_loss = total_reg_loss / n_batches

    # Compute summary classification metrics
    if cls_preds_all:
        from sklearn.metrics import accuracy_score, roc_auc_score

        cls_preds = torch.cat(cls_preds_all, dim=0).numpy()
        cls_tgts = torch.cat(cls_targets_all, dim=0).numpy()
        cls_masks = torch.cat(cls_masks_all, dim=0).numpy()

        accs = []
        aucs = []
        for t in range(cls_preds.shape[1]):
            mask = cls_masks[:, t].astype(bool)
            if mask.sum() == 0:
                continue
            y_score = cls_preds[mask, t]
            y_true = cls_tgts[mask, t].astype(int)
            # Binary thresholding for accuracy
            y_pred = (y_score > 0.5).astype(int)
            try:
                accs.append(accuracy_score(y_true, y_pred))
                if len(np.unique(y_true)) > 1:
                    aucs.append(roc_auc_score(y_true, y_score))
            except Exception:
                # Skip degenerate tasks
                continue

        if accs:
            metrics['avg_cls_accuracy'] = float(np.mean(accs))
        if aucs:
            metrics['avg_auc'] = float(np.mean(aucs))

    # Compute summary regression metrics
    if reg_preds_all:
        from sklearn.metrics import mean_squared_error, r2_score

        r_preds = torch.cat(reg_preds_all, dim=0).numpy()
        r_tgts = torch.cat(reg_targets_all, dim=0).numpy()
        r_masks = torch.cat(reg_masks_all, dim=0).numpy()

        r2s = []
        rmses = []
        for t in range(r_preds.shape[1]):
            mask = r_masks[:, t].astype(bool)
            if mask.sum() == 0:
                continue
            y_pred = r_preds[mask, t]
            y_true = r_tgts[mask, t]
            try:
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)  # Manual square root for older sklearn versions
                # Guard against zero variance targets
                r2 = 0.0 if np.var(y_true) < 1e-6 else r2_score(y_true, y_pred)
                r2s.append(r2)
                rmses.append(rmse)
            except Exception:
                continue

        if r2s:
            metrics['avg_r2'] = float(np.mean(r2s))
        if rmses:
            metrics['avg_rmse'] = float(np.mean(rmses))

    return metrics, avg_loss, avg_cls_loss, avg_reg_loss


def find_best_checkpoint(exp_dir: Path) -> Optional[Path]:
    ckpt_dir = exp_dir / 'checkpoints'
    if not ckpt_dir.exists():
        return None
    cands = list(ckpt_dir.glob('*_best.pth'))
    if cands:
        # Prefer the first matching *_best.pth
        return cands[0]
    # Fallback: pick most recently modified .pth
    all_ckpts = sorted(ckpt_dir.glob('*.pth'), key=lambda p: p.stat().st_mtime, reverse=True)
    return all_ckpts[0] if all_ckpts else None


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate ToxD4C checkpoint on test set')
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='Path to experiment directory under ToxD4C/experiments/...')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth; overrides --experiment_dir if set')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='LMDB directory containing train.lmdb/valid.lmdb/test.lmdb')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_atoms', type=int, default=64)

    args = parser.parse_args()

    # Resolve checkpoint and config
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        ckpt_dir = ckpt_path.parent
    else:
        if not args.experiment_dir:
            raise ValueError('Either --checkpoint or --experiment_dir must be provided')
        exp_dir = Path(args.experiment_dir)
        ckpt_path = find_best_checkpoint(exp_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f'No checkpoint found under: {exp_dir}/checkpoints')
        ckpt_dir = ckpt_path.parent

    config_path = ckpt_dir / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path} (expected next to checkpoint)')

    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Checkpoint: {ckpt_path}')
    logger.info(f'Data dir: {args.data_dir}')

    # Build model and load weights
    model = ToxD4C(config=config, device=device).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    _, _, test_loader = create_lmdb_dataloaders(args.data_dir, batch_size=args.batch_size, max_atoms=args.max_atoms)

    # Evaluate
    metrics, total_loss, cls_loss, reg_loss = evaluate_on_loader(model, test_loader, device)

    # Log to console
    print('Final Test Results:')
    print(f'  Total Loss: {total_loss:.4f}')
    print(f'  Classification Loss: {cls_loss:.4f}')
    print(f'  Regression Loss: {reg_loss:.4f}')
    if metrics:
        if 'avg_cls_accuracy' in metrics:
            print(f"  Average Classification Accuracy: {metrics['avg_cls_accuracy']:.4f}")
        if 'avg_auc' in metrics:
            print(f"  Average AUC: {metrics['avg_auc']:.4f}")
        if 'avg_r2' in metrics:
            print(f"  Average R²: {metrics['avg_r2']:.4f}")
        if 'avg_rmse' in metrics:
            print(f"  Average RMSE: {metrics['avg_rmse']:.4f}")

    # Persist JSON next to checkpoint
    out = {
        'split': 'test',
        'checkpoint': str(ckpt_path),
        'metrics': metrics,
        'total_loss': total_loss,
        'classification_loss': cls_loss,
        'regression_loss': reg_loss,
    }
    out_path = ckpt_dir / 'test_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f'Test results saved to: {out_path}')


if __name__ == '__main__':
    main()

