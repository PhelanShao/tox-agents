import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Simple fusion module using feature concatenation."""

    def __init__(self, dims, out_dim):
        super().__init__()
        self.proj = nn.Linear(sum(dims), out_dim)

    def forward(self, xs):
        """Concatenate inputs and project to ``out_dim``."""
        return self.proj(torch.cat(xs, dim=-1))


class CrossAttnFusion(nn.Module):
    """Cross-attention based fusion with a learnable gating mechanism."""

    def __init__(self, d_q, d_kv, nhead: int = 4, out_dim: int = 256):
        super().__init__()
        self.q = nn.Linear(d_q, out_dim)
        self.kv = nn.Linear(d_kv, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, nhead, batch_first=True)
        self.out = nn.Linear(out_dim, out_dim)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_q, x_kv):
        q = self.q(x_q).unsqueeze(1)
        kv = self.kv(x_kv).unsqueeze(1)
        y, _ = self.attn(q, kv, kv)
        y = y.squeeze(1)
        return self.out(self.gate * y + (1 - self.gate) * x_q)
