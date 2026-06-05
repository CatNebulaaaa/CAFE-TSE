from __future__ import annotations

import torch


def stft(wav: torch.Tensor, n_fft: int, hop_length: int, win_length: int | None = None) -> torch.Tensor:
    window = torch.hann_window(win_length or n_fft, device=wav.device, dtype=wav.dtype)
    return torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length or n_fft, window=window, return_complex=True)


def istft(spec: torch.Tensor, n_fft: int, hop_length: int, length: int, win_length: int | None = None) -> torch.Tensor:
    window = torch.hann_window(win_length or n_fft, device=spec.device, dtype=spec.real.dtype)
    return torch.istft(spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length or n_fft, window=window, length=length)

