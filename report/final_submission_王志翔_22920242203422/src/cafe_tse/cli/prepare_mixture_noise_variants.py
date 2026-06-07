from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cafe_tse.utils.audio_io import fix_length, read_wav, write_wav
from cafe_tse.utils.manifest import read_manifest, write_manifest


def _resolve(manifest_path: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    return (manifest_path.parent / path).resolve()


def _scale_to_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = torch.mean(signal**2).clamp_min(1e-8)
    noise_power = torch.mean(noise**2).clamp_min(1e-8)
    return noise * torch.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10.0))))


def _white_noise_like(wav: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(wav.shape, generator=gen, dtype=wav.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--snrs", nargs="+", type=float, default=[5.0, 10.0])
    parser.add_argument("--noise_types", nargs="+", default=["babble"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    out_dir = Path(args.out_dir)
    audio_root = out_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    variant_index = []

    for noise_type in args.noise_types:
        for snr in args.snrs:
            snr_tag = f"{snr:g}db"
            variant = f"mixture_{noise_type}_{snr_tag}"
            variant_audio_dir = audio_root / variant
            variant_audio_dir.mkdir(parents=True, exist_ok=True)
            variant_rows = []
            for idx, row in enumerate(rows):
                mix_path = _resolve(manifest_path, row["mixture_path"])
                mixture, _ = read_wav(mix_path, target_sr=args.sample_rate)
                if noise_type == "white":
                    noise = _white_noise_like(mixture, args.seed + idx)
                elif noise_type == "babble":
                    noise_row = rows[(idx * 17 + 13) % len(rows)]
                    noise_path = _resolve(manifest_path, noise_row["target_path"])
                    noise, _ = read_wav(noise_path, target_sr=args.sample_rate)
                    noise = fix_length(noise, mixture.numel())
                else:
                    raise ValueError(f"Unsupported noise type: {noise_type}")

                noisy = torch.clamp(mixture + _scale_to_snr(mixture, noise, snr), -1.0, 1.0)
                out_wav = variant_audio_dir / f"{row['utt_id']}.wav"
                write_wav(out_wav, noisy, args.sample_rate)

                new_row = dict(row)
                new_row["mixture_path"] = str(out_wav)
                new_row["mixture_noise"] = noise_type
                new_row["mixture_noise_snr"] = snr
                variant_rows.append(new_row)

            out_manifest = out_dir / f"{variant}.csv"
            write_manifest(variant_rows, out_manifest)
            variant_index.append(
                {
                    "variant": variant,
                    "manifest": str(out_manifest),
                    "noise_type": noise_type,
                    "snr": snr,
                }
            )

    with (out_dir / "variants.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "manifest", "noise_type", "snr"])
        writer.writeheader()
        writer.writerows(variant_index)
    print(f"wrote {len(variant_index)} mixture-noise variants to {out_dir}")


if __name__ == "__main__":
    main()
