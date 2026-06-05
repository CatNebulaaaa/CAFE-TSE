from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def read_wav(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[torch.Tensor, int]:
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except ModuleNotFoundError:
        from scipy.io import wavfile

        sr, wav_raw = wavfile.read(str(path))
        wav = wav_raw.astype(np.float32)
        if np.issubdtype(wav_raw.dtype, np.integer):
            wav = wav / max(float(np.iinfo(wav_raw.dtype).max), 1.0)
    if wav.ndim == 2 and mono:
        wav = wav.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        try:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        except Exception as exc:
            raise RuntimeError(f"Resampling {path} from {sr} to {target_sr} requires librosa") from exc
    return torch.from_numpy(np.asarray(wav, dtype=np.float32)), sr


def write_wav(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = wav.detach().cpu().float().numpy()
    if data.ndim > 1:
        data = data.squeeze()
    data = np.clip(data, -1.0, 1.0)
    try:
        import soundfile as sf

        sf.write(str(path), data, sample_rate)
    except ModuleNotFoundError:
        from scipy.io import wavfile

        wavfile.write(str(path), sample_rate, (data * 32767.0).astype(np.int16))


def fix_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.numel() >= length:
        return wav[:length]
    return torch.nn.functional.pad(wav, (0, length - wav.numel()))


def rms_normalize(wav: torch.Tensor, target: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(eps)
    return wav * (target / rms)
