from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cafe_tse.utils.audio_io import fix_length, read_wav, write_wav
from cafe_tse.utils.manifest import read_manifest, write_manifest


def _snr_noise(wav: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(wav.shape, generator=gen, dtype=wav.dtype)
    signal_power = torch.mean(wav**2).clamp_min(1e-8)
    noise_power = torch.mean(noise**2).clamp_min(1e-8)
    scale = torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10.0))))
    return torch.clamp(wav + noise * scale, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--durations", nargs="+", type=float, default=[1.0, 3.0, 5.0])
    parser.add_argument("--noise_snr_db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    audio_root = out_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    variant_index = []

    for duration in args.durations:
        samples = int(args.sample_rate * duration)
        for noise_label in ["clean", f"noisy_{int(args.noise_snr_db)}db"]:
            variant = f"enroll_{duration:g}s_{noise_label}"
            variant_rows = []
            variant_audio_dir = audio_root / variant
            variant_audio_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in enumerate(rows):
                src = Path(str(row["enrollment_path"]))
                if not src.is_absolute() and not src.exists():
                    src = (manifest_path.parent / src).resolve()
                wav, _ = read_wav(src, target_sr=args.sample_rate)
                wav = fix_length(wav, samples)
                if noise_label != "clean":
                    wav = _snr_noise(wav, args.noise_snr_db, args.seed + idx)
                out_wav = variant_audio_dir / f"{row['utt_id']}.wav"
                write_wav(out_wav, wav, args.sample_rate)

                new_row = dict(row)
                new_row["enrollment_path"] = str(out_wav)
                new_row["enrollment_length"] = duration
                new_row["enrollment_noise"] = noise_label
                variant_rows.append(new_row)

            out_manifest = out_dir / f"{variant}.csv"
            write_manifest(variant_rows, out_manifest)
            variant_index.append({"variant": variant, "manifest": str(out_manifest), "duration": duration, "noise": noise_label})

    with (out_dir / "variants.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "manifest", "duration", "noise"])
        writer.writeheader()
        writer.writerows(variant_index)
    print(f"wrote {len(variant_index)} enrollment variants to {out_dir}")


if __name__ == "__main__":
    main()
