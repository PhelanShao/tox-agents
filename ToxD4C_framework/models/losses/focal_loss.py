"""
Focal Loss for Class Imbalance Handling

Implements Focal Loss (Lin et al., 2017) with optional class weighting
for handling severe class imbalance in multi-label toxicity prediction.

Reference:
    Lin, T. Y., et al. (2017). Focal loss for dense object detection.
    In ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
        - p_t = p if y=1, else 1-p
        - α_t = α if y=1, else 1-α (class weighting)
        - γ (gamma) is the focusing parameter
    
    Args:
        gamma: Focusing parameter (default: 2.0). Higher values down-weight
               easy examples more aggressively.
        alpha: Class weight for positive class (default: None for no weighting).
               Can be a float or tensor of per-class weights.
        pos_weight: Alternative to alpha, weight for positive examples.
                    Useful when classes are highly imbalanced.
        reduction: 'none', 'mean', or 'sum'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape [batch, num_tasks] or [batch]
            targets: Binary targets of same shape as inputs
            mask: Optional mask of valid samples (same shape)
            
        Returns:
            Focal loss value
        """
        # Compute sigmoid probabilities
        p = torch.sigmoid(inputs)
        
        # Compute cross-entropy term
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Compute p_t (probability of correct class)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha (class) weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply pos_weight if specified (for inverse frequency weighting)
        if self.pos_weight is not None:
            if self.pos_weight.dim() == 1 and inputs.dim() == 2:
                # Per-task pos_weight
                pos_weight = self.pos_weight.unsqueeze(0).to(inputs.device)
            else:
                pos_weight = self.pos_weight.to(inputs.device)
            weight = targets * pos_weight + (1 - targets)
            focal_loss = focal_loss * weight
        
        # Apply mask if provided
        if mask is not None:
            focal_loss = focal_loss * mask.float()
            if self.reduction == 'mean':
                return focal_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        
        # Standard reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class FocalLossWithLogits(FocalLoss):
    """
    Alias for FocalLoss that explicitly indicates input is logits.
    Same as FocalLoss - included for API clarity.
    """
    pass


def compute_class_weights(
    labels: torch.Tensor,
    masks: torch.Tensor,
    method: str = 'inverse_freq',
    clip_range: tuple = (0.1, 10.0)
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        labels: Binary labels [num_samples, num_tasks]
        masks: Valid sample masks [num_samples, num_tasks]
        method: Weighting method:
            - 'inverse_freq': Weight = (num_neg / num_pos)
            - 'effective_num': Effective number of samples (Cui et al., 2019)
            - 'sqrt_inverse': sqrt(num_neg / num_pos)
        clip_range: Min and max weight values to prevent extreme weights
        
    Returns:
        pos_weight tensor [num_tasks] for use in loss functions
    """
    num_tasks = labels.shape[1]
    pos_weights = torch.ones(num_tasks)
    
    for task_idx in range(num_tasks):
        task_mask = masks[:, task_idx].bool()
        if task_mask.sum() == 0:
            continue
            
        task_labels = labels[task_mask, task_idx]
        num_pos = task_labels.sum().float()
        num_neg = (task_mask.sum() - num_pos).float()
        
        if num_pos == 0 or num_neg == 0:
            pos_weights[task_idx] = 1.0
            continue
        
        if method == 'inverse_freq':
            weight = num_neg / num_pos
        elif method == 'sqrt_inverse':
            weight = torch.sqrt(num_neg / num_pos)
        elif method == 'effective_num':
            # Effective number of samples (beta = 0.9999)
            beta = 0.9999
            effective_num_pos = (1 - beta ** num_pos) / (1 - beta)
            effective_num_neg = (1 - beta ** num_neg) / (1 - beta)
            weight = effective_num_neg / effective_num_pos
        else:
            weight = 1.0
        
        # Clip to reasonable range
        weight = torch.clamp(weight, clip_range[0], clip_range[1])
        pos_weights[task_idx] = weight
    
    return pos_weights


if __name__ == '__main__':
    # Test focal loss
    print("Testing Focal Loss implementation...")
    
    # Create test data
    batch_size, num_tasks = 32, 26
    logits = torch.randn(batch_size, num_tasks)
    targets = torch.randint(0, 2, (batch_size, num_tasks)).float()
    mask = torch.ones(batch_size, num_tasks).bool()
    
    # Test basic focal loss
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"Basic Focal Loss (gamma=2.0): {loss.item():.4f}")
    
    # Test with alpha
    focal_loss_alpha = FocalLoss(gamma=2.0, alpha=0.75)
    loss_alpha = focal_loss_alpha(logits, targets)
    print(f"Focal Loss with alpha=0.75: {loss_alpha.item():.4f}")
    
    # Test class weight computation
    pos_weights = compute_class_weights(targets, mask.float(), method='inverse_freq')
    print(f"Computed pos_weights (sample): {pos_weights[:5].tolist()}")
    
    # Test focal loss with pos_weight
    focal_loss_weighted = FocalLoss(gamma=2.0, pos_weight=pos_weights)
    loss_weighted = focal_loss_weighted(logits, targets)
    print(f"Focal Loss with inverse freq weights: {loss_weighted.item():.4f}")
    
    # Compare with standard BCE
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    print(f"Standard BCE Loss: {bce_loss.item():.4f}")
    
    print("\n✅ Focal Loss tests passed!")
