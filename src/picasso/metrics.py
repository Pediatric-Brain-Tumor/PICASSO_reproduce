"""Streaming reconstruction / sparsity metrics for an SAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Accumulator:
    def __init__(self, d_sae: int, d: int, device: str = "cpu"):
        self.n = 0
        self.sum_x = torch.zeros(d, dtype=torch.float64, device=device)
        self.sum_x2 = torch.zeros(d, dtype=torch.float64, device=device)
        self.sse = 0.0
        self.cos = 0.0
        self.l0 = 0.0
        self.ev_ps = 0.0
        self.fired = torch.zeros(d_sae, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor) -> None:
        self.n += len(x)
        self.sum_x += x.sum(0, dtype=torch.float64)
        self.sum_x2 += x.double().pow(2).sum(0)
        self.sse += (x_hat - x).double().pow(2).sum().item()
        self.cos += F.cosine_similarity(x, x_hat, dim=1).double().sum().item()
        # SAELens-style per-sample explained variance: 1 - var(resid)/var(x) across features
        resid = x_hat - x
        self.ev_ps += (1 - resid.var(dim=1) / x.var(dim=1).clamp_min(1e-12)).double().sum().item()
        active = f != 0
        self.l0 += active.sum().item()
        self.fired += active.sum(0)

    def summary(self) -> dict[str, float]:
        n = self.n
        total_var = (self.sum_x2 - self.sum_x.pow(2) / n).sum().item()   # sum over dims of n*var
        freq = self.fired.double() / n
        return {
            "n": n,
            "mse": self.sse / (n * len(self.sum_x)),
            "normalized_mse": self.sse / total_var,
            "explained_variance": 1 - self.sse / total_var,       # 1 - FVU (global, Fig 2a style)
            "explained_variance_per_sample": self.ev_ps / n,
            "cosine": self.cos / n,
            "l0": self.l0 / n,
            "n_dead": int((freq == 0).sum()),
            "n_dense": int((freq > 1e-2).sum()),
            "n_active_1e-3": int((freq > 1e-3).sum()),            # paper: ~10,000 concepts with freq > 0.1%
            "n_active_1e-5": int((freq > 1e-5).sum()),
        }

    def frequencies(self) -> torch.Tensor:
        return (self.fired.double() / max(self.n, 1)).cpu()
