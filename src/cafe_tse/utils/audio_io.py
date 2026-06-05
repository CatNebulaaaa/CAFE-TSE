from __future__ import annotations

from pathlib import Path
import wave

import numpy as np
import torch


def _read_wav_stdlib(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sr = handle.getframerate()
        n_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width {sample_width} bytes in {path}")
    wav = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if n_channels > 1:
        wav = wav.reshape(-1, n_channels)
    return wav, sr


def _write_wav_stdlib(path: str | Path, data: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    n_channels = 1
    if pcm.ndim == 2:
        n_channels = pcm.shape[1]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(n_channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_wav(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[torch.Tensor, int]:
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except ModuleNotFoundError:
        try:
            from scipy.io import wavfile

            sr, wav_raw = wavfile.read(str(path))
            wav = wav_raw.astype(np.float32)
            if np.issubdtype(wav_raw.dtype, np.integer):
                wav = wav / max(float(np.iinfo(wav_raw.dtype).max), 1.0)
        except Exception:
            wav, sr = _read_wav_stdlib(path)
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
        try:
            from scipy.io import wavfile

            wavfile.write(str(path), sample_rate, (data * 32767.0).astype(np.int16))
        except Exception:
            _write_wav_stdlib(path, data, sample_rate)


def fix_length(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.numel() >= length:
        return wav[:length]
    return torch.nn.functional.pad(wav, (0, length - wav.numel()))


def rms_normalize(wav: torch.Tensor, target: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(eps)
    return wav * (target / rms)
