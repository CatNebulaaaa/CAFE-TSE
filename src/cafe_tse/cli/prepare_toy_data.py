from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from cafe_tse.utils.audio_io import write_wav
from cafe_tse.utils.manifest import write_manifest


def _sine(freq: float, t: np.ndarray, phase: float) -> np.ndarray:
    return np.sin(2 * math.pi * freq * t + phase).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/toy")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    mix_dir = out / "mixtures"
    src_dir = out / "targets"
    enr_dir = out / "enrollments"
    for d in [mix_dir, src_dir, enr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    n = int(args.sample_rate * args.duration)
    t = np.arange(n, dtype=np.float32) / args.sample_rate
    rows = []
    for i in range(args.num_samples):
        target_freq = 180 + 13 * (i % 7)
        interferer_freq = 310 + 17 * (i % 9)
        target = 0.35 * _sine(target_freq, t, rng.uniform(0, 2 * math.pi))
        interferer = 0.28 * _sine(interferer_freq, t, rng.uniform(0, 2 * math.pi))
        envelope = 0.5 + 0.5 * _sine(1.0 + 0.1 * (i % 3), t, 0)
        target = target * (0.7 + 0.3 * envelope)
        interferer = interferer * (0.4 + 0.6 * envelope[::-1])
        noise = rng.normal(0.0, 0.01, size=n).astype(np.float32)
        mixture = target + interferer + noise
        enrollment = 0.35 * _sine(target_freq, t, rng.uniform(0, 2 * math.pi))

        utt = f"toy_{i:03d}"
        mix_path = mix_dir / f"{utt}.wav"
        tgt_path = src_dir / f"{utt}.wav"
        enr_path = enr_dir / f"{utt}.wav"
        write_wav(mix_path, torch_from_np(mixture), args.sample_rate)
        write_wav(tgt_path, torch_from_np(target), args.sample_rate)
        write_wav(enr_path, torch_from_np(enrollment), args.sample_rate)
        difficulty = ["easy", "medium", "hard"][i % 3]
        rows.append(
            {
                "utt_id": utt,
                "mixture_path": str(mix_path),
                "target_path": str(tgt_path),
                "enrollment_path": str(enr_path),
                "speaker_id": f"spk_{i % 4}",
                "split": "toy",
                "sample_rate": args.sample_rate,
                "duration": args.duration,
                "num_speakers": args.num_speakers,
                "sir": [5, 0, -5][i % 3],
                "snr": 20,
                "overlap_ratio": 0.7,
                "gender_condition": "mixed",
                "enrollment_length": args.duration,
                "enrollment_noise": "clean",
                "difficulty": difficulty,
                "complexity_score": 0.5,
            }
        )
    write_manifest(rows, out / "toy_manifest.csv")
    print(f"wrote {out / 'toy_manifest.csv'}")


def torch_from_np(array: np.ndarray):
    import torch

    return torch.from_numpy(array.astype(np.float32))


if __name__ == "__main__":
    main()
