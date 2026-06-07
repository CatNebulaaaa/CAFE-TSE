from __future__ import annotations

import torch


def cosine_wave_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.numel(), b.numel())
    if n == 0:
        return float("nan")
    return torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0).item()

