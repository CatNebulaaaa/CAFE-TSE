from __future__ import annotations

import torch
import torch.nn as nn


class SparseConditionFusion(nn.Module):
    """FiLM-style target condition injection for selected separator blocks."""

    def __init__(self, dim: int, condition_dim: int, mode: str = "film"):
        super().__init__()
        self.mode = mode
        self.to_gamma = nn.Linear(condition_dim, dim)
        self.to_beta = nn.Linear(condition_dim, dim)
        if mode == "gated":
            self.to_gate = nn.Linear(dim + condition_dim, dim)
        elif mode != "film":
            raise ValueError(f"unsupported condition fusion mode: {mode}")
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        gamma = self.to_gamma(condition).unsqueeze(1).unsqueeze(1)
        beta = self.to_beta(condition).unsqueeze(1).unsqueeze(1)
        delta = gamma * x_norm + beta
        if self.mode == "gated":
            cond = condition.unsqueeze(1).unsqueeze(1).expand(-1, x.shape[1], x.shape[2], -1)
            gate = torch.sigmoid(self.to_gate(torch.cat([x_norm, cond], dim=-1)))
            delta = gate * delta
        return x + delta
