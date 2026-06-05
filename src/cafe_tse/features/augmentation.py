from __future__ import annotations

import torch


def add_noise(wav: torch.Tensor, snr_db: float, generator: torch.Generator | None = None) -> torch.Tensor:
    noise = torch.randn(wav.shape, generator=generator, device=wav.device, dtype=wav.dtype)
    sig_power = wav.pow(2).mean().clamp_min(1e-8)
    noise_power = noise.pow(2).mean().clamp_min(1e-8)
    scale = (sig_power / (10 ** (snr_db / 10.0)) / noise_power).sqrt()
    return wav + noise * scale

