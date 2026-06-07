from __future__ import annotations

import torch
import torch.nn as nn


class DynamicRouter(nn.Module):
    """Rule-based route selector for explainable dynamic inference."""

    def __init__(self, threshold_easy: float = 0.35, threshold_hard: float = 0.65):
        super().__init__()
        self.threshold_easy = threshold_easy
        self.threshold_hard = threshold_hard

    def forward(self, complexity_score: torch.Tensor) -> list[str]:
        routes = []
        for score in complexity_score.detach().cpu().view(-1).tolist():
            if score < self.threshold_easy:
                routes.append("shallow")
            elif score < self.threshold_hard:
                routes.append("lite")
            else:
                routes.append("full")
        return routes

    def active_blocks(self, routes: list[str], shallow_blocks: int, lite_blocks: int, full_blocks: int) -> list[int]:
        out = []
        for route in routes:
            if route == "shallow":
                out.append(shallow_blocks)
            elif route == "lite":
                out.append(lite_blocks)
            else:
                out.append(full_blocks)
        return out

