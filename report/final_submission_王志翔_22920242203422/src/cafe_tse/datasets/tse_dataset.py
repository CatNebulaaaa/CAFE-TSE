from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from cafe_tse.utils.audio_io import fix_length, read_wav, rms_normalize
from cafe_tse.utils.manifest import read_manifest


def _rms_gain(wav: torch.Tensor, target: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(eps)
    return target / rms


class TSEDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        sample_rate: int,
        segment_seconds: float,
        normalize_audio: bool = True,
    ):
        self.manifest_path = Path(manifest)
        self.rows = read_manifest(manifest)
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.normalize_audio = normalize_audio
        self.allowed_difficulties: set[str] | None = None

    def set_allowed_difficulties(self, difficulties: list[str] | None) -> None:
        self.allowed_difficulties = set(difficulties) if difficulties else None

    def _visible_rows(self) -> list[dict]:
        if not self.allowed_difficulties:
            return self.rows
        filtered = [r for r in self.rows if r.get("difficulty") in self.allowed_difficulties]
        return filtered if filtered else self.rows

    def __len__(self) -> int:
        return len(self._visible_rows())

    def _resolve(self, value: str) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (self.manifest_path.parent / path).resolve() if not path.exists() else path

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self._visible_rows()[idx]
        mixture, _ = read_wav(self._resolve(row["mixture_path"]), target_sr=self.sample_rate)
        target, _ = read_wav(self._resolve(row["target_path"]), target_sr=self.sample_rate)
        enrollment, _ = read_wav(self._resolve(row["enrollment_path"]), target_sr=self.sample_rate)

        mixture = fix_length(mixture, self.segment_samples)
        target = fix_length(target, self.segment_samples)
        enrollment = fix_length(enrollment, self.segment_samples)
        norm_gain = torch.tensor(1.0, dtype=mixture.dtype)
        if self.normalize_audio:
            norm_gain = _rms_gain(mixture).to(dtype=mixture.dtype)
            mixture = mixture * norm_gain
            target = target * norm_gain
            enrollment = rms_normalize(enrollment)

        return {
            "utt_id": str(row["utt_id"]),
            "mixture": mixture.float(),
            "target": target.float(),
            "enrollment": enrollment.float(),
            "complexity_score": torch.tensor(float(row.get("complexity_score", 0.5)), dtype=torch.float32),
            "difficulty": str(row.get("difficulty", "medium")),
            "sample_rate": int(row.get("sample_rate", self.sample_rate) or self.sample_rate),
            "norm_gain": norm_gain.float(),
        }
