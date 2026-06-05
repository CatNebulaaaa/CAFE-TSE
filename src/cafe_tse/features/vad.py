from __future__ import annotations

import torch

from cafe_tse.features.complexity import vad_ratio


def energy_vad(wav: torch.Tensor, frame: int = 512, hop: int = 128, threshold: float = 1e-4) -> torch.Tensor:
    return vad_ratio(wav, frame=frame, hop=hop, threshold=threshold)

