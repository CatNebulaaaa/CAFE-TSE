from __future__ import annotations

import torch


def spectral_l1_loss(est: torch.Tensor, target: torch.Tensor, n_fft: int = 512, hop_length: int = 128) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=est.device, dtype=est.dtype)
    est_spec = torch.stft(est, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)
    tgt_spec = torch.stft(target, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)
    return (est_spec.abs() - tgt_spec.abs()).abs().mean()


def multi_resolution_spectral_l1_loss(
    est: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: list[int] | tuple[int, ...] = (256, 512, 1024),
    hop_ratio: float = 0.25,
) -> torch.Tensor:
    losses = []
    for n_fft in fft_sizes:
        hop_length = max(1, int(n_fft * hop_ratio))
        losses.append(spectral_l1_loss(est, target, n_fft=int(n_fft), hop_length=hop_length))
    return torch.stack(losses).mean()
