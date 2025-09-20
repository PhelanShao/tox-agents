import torch
import torch.nn as nn
from typing import Dict, List, Optional


class HomoscedasticWeighting(nn.Module):
    """Homoscedastic uncertainty-based task weighting (Kendall & Gal, 2018).

    L_total = sum_i( exp(-s_i) * L_i + s_i ), where s_i = log(sigma_i^2) are learned.
    This form avoids explicit 1/2 factors and works for both regression and classification losses.
    """

    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        # Initialize log variances to 0 => sigma^2=1 (neutral weighting)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for name, loss in losses.items():
            if loss is None:
                continue
            # s = log(sigma^2)
            s = self.log_vars[name]
            total = total + torch.exp(-s) * loss + s
        return total

    def get_sigmas(self) -> Dict[str, float]:
        return {k: float(torch.exp(0.5 * v).detach().cpu()) for k, v in self.log_vars.items()}


class GradNormWeighting(nn.Module):
    """GradNorm: Gradient normalization for adaptive multi-task weighting (Chen et al., 2018).

    Maintains positive task weights w_i and updates them to balance gradient norms across tasks.
    Usage pattern (each step):
      1) Compute per-task losses and weighted total L = sum_i w_i * L_i
      2) Backprop L and measure per-task gradient norms on a shared layer
      3) Call update(g_norms, L_i(t), L_i(0)) to update w_i
    """

    def __init__(self, task_names: List[str], alpha: float = 0.5):
        super().__init__()
        self.task_names = task_names
        self.alpha = alpha
        # Initialize weights to 1.0 and make them learnable
        self.log_w = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })
        self._initial_losses: Dict[str, Optional[float]] = {name: None for name in task_names}

    def weights(self) -> Dict[str, torch.Tensor]:
        return {k: torch.exp(v) for k, v in self.log_w.items()}

    def weighted_sum(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for name, loss in losses.items():
            if loss is None:
                continue
            total = total + torch.exp(self.log_w[name]) * loss
        return total

    def set_initial_losses(self, init_losses: Dict[str, float]):
        for k, v in init_losses.items():
            self._initial_losses[k] = float(v)

    @torch.no_grad()
    def update(self, grad_norms: Dict[str, float], current_losses: Dict[str, float]):
        # Compute task loss ratios r_i = (L_i(t) / L_i(0))^alpha
        ratios: Dict[str, float] = {}
        for name, cur in current_losses.items():
            L0 = self._initial_losses.get(name, None)
            if L0 is None or L0 == 0:
                ratios[name] = 1.0
            else:
                ratios[name] = float((cur / L0) ** self.alpha)

        # Target gradient norm: 
        #   G_avg * r_i where G_avg = mean_j(G_j)
        G_vals = [float(g) for g in grad_norms.values() if g is not None]
        if len(G_vals) == 0:
            return
        G_avg = sum(G_vals) / len(G_vals)

        # Gradient descent step on |G_i - G_avg * r_i|
        # Here we perform a simple proportional update in log-weight space.
        eta = 0.025  # small step size for stable updates
        for name in self.task_names:
            G_i = float(grad_norms.get(name, 0.0))
            target = G_avg * ratios.get(name, 1.0)
            # d/d log w_i proportional to (G_i - target)
            delta = eta * (G_i - target)
            self.log_w[name].data = self.log_w[name].data - delta

    def get_weights(self) -> Dict[str, float]:
        return {k: float(torch.exp(v).detach().cpu()) for k, v in self.log_w.items()}


__all__ = [
    "HomoscedasticWeighting",
    "GradNormWeighting",
]

