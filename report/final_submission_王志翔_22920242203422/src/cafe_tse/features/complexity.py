from __future__ import annotations

import torch


def spectral_entropy(magnitude: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mag = magnitude.clamp_min(eps)
    prob = mag / mag.sum(dim=-2, keepdim=True).clamp_min(eps)
    ent = -(prob * prob.log()).sum(dim=-2)
    denom = torch.log(torch.tensor(magnitude.shape[-2], device=magnitude.device, dtype=magnitude.dtype))
    return (ent / denom.clamp_min(eps)).mean(dim=-1)


def _frames(wav: torch.Tensor, frame: int, hop: int) -> torch.Tensor:
    if wav.shape[-1] < frame:
        wav = torch.nn.functional.pad(wav, (0, frame - wav.shape[-1]))
    return wav.unfold(-1, frame, hop)


def energy_variance(wav: torch.Tensor, frame: int = 512, hop: int = 128, eps: float = 1e-8) -> torch.Tensor:
    frames = _frames(wav, frame, hop)
    energy = frames.pow(2).mean(dim=-1)
    return energy.var(dim=-1, unbiased=False) / energy.mean(dim=-1).clamp_min(eps)


def vad_ratio(wav: torch.Tensor, frame: int = 512, hop: int = 128, threshold: float = 1e-4) -> torch.Tensor:
    frames = _frames(wav, frame, hop)
    energy = frames.pow(2).mean(dim=-1)
    return (energy > threshold).float().mean(dim=-1)


def spectral_flatness(magnitude: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mag = magnitude.clamp_min(eps)
    geo = mag.log().mean(dim=-2).exp()
    arith = mag.mean(dim=-2).clamp_min(eps)
    return (geo / arith).mean(dim=-1)


def compute_complexity_score(
    wav: torch.Tensor,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 128,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    del sample_rate
    if weights is None:
        weights = {"entropy": 0.35, "vad": 0.25, "energy": 0.25, "flatness": 0.15}
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    mag = spec.abs()
    score = (
        weights["entropy"] * spectral_entropy(mag)
        + weights["vad"] * vad_ratio(wav, frame=n_fft, hop=hop_length)
        + weights["energy"] * torch.tanh(energy_variance(wav, frame=n_fft, hop=hop_length))
        + weights["flatness"] * spectral_flatness(mag)
    )
    return score.clamp(0.0, 1.0)

