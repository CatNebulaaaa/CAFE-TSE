from __future__ import annotations

import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def skip_ratio(active_blocks: list[int], full_blocks: int) -> float:
    if not active_blocks:
        return 0.0
    return 1.0 - (sum(active_blocks) / (len(active_blocks) * max(full_blocks, 1)))

