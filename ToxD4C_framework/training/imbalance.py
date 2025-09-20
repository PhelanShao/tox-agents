import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor,
               targets: torch.Tensor,
               alpha: float = 0.25,
               gamma: float = 2.0,
               reduction: str = 'mean') -> torch.Tensor:
    """Compute the focal loss for binary classification.

    Parameters
    ----------
    logits: torch.Tensor
        Raw model outputs of shape ``(N,)`` or ``(N, 1)``.
    targets: torch.Tensor
        Binary ground truth labels with the same shape as ``logits``.
    alpha: float, default 0.25
        Balancing factor between positive and negative samples.
    gamma: float, default 2.0
        Focusing parameter controlling down-weighting of easy examples.
    reduction: str, default "mean"
        Specifies the reduction to apply to the output: ``'none'``,
        ``'mean'`` or ``'sum'``.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = (alpha * (1 - p_t) ** gamma) * ce
    if reduction == 'sum':
        return loss.sum()
    if reduction == 'mean':
        return loss.mean()
    return loss
