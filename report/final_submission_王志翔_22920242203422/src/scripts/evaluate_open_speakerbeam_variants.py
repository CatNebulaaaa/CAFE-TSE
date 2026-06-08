from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "asteroid_site"))
sys.path.insert(0, str(ROOT / "third_party" / "speakerbeam" / "src"))

from train_open_speakerbeam import build_model  # noqa: E402

from cafe_tse.datasets.collate import tse_collate  # noqa: E402
from cafe_tse.datasets.tse_dataset import TSEDataset  # noqa: E402
from cafe_tse.metrics.separation import compute_bss_metrics  # noqa: E402
from cafe_tse.utils.audio_io import fix_length, read_wav  # noqa: E402


def egsp_waveform(
    mixture: torch.Tensor,
    enrollment: torch.Tensor,
    strength: float,
    min_weight: float,
    max_weight: float,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    if strength == 0:
        return mixture
    window = torch.hann_window(n_fft, device=mixture.device)
    mix_spec = torch.stft(mixture, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    enroll_spec = torch.stft(enrollment, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    profile = enroll_spec.abs().mean(dim=-1)
    profile = profile / profile.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    weight = (1.0 + strength * (profile - 1.0)).clamp(min_weight, max_weight)
    enhanced = torch.istft(mix_spec * weight.unsqueeze(-1), n_fft=n_fft, hop_length=hop_length, window=window, length=mixture.shape[-1])
    return enhanced


def resolve_audio(ds: TSEDataset, row: dict[str, str], key: str, sample_rate: int, length: int) -> torch.Tensor:
    wav, _ = read_wav(ds._resolve(str(row[key])), target_sr=sample_rate)
    return fix_length(wav, length)


def make_enrollment(
    ds: TSEDataset,
    row: dict[str, str],
    variant: str,
    sample_rate: int,
    length: int,
    norm_gain: torch.Tensor,
    shuffled_row: dict[str, str] | None,
    generator: torch.Generator,
) -> torch.Tensor:
    if variant == "correct":
        wav = resolve_audio(ds, row, "enrollment_path", sample_rate, length)
    elif variant == "shuffled":
        if shuffled_row is None:
            raise ValueError("shuffled variant requires shuffled_row")
        wav = resolve_audio(ds, shuffled_row, "enrollment_path", sample_rate, length)
    elif variant == "interferer":
        wav = resolve_audio(ds, row, "interferer_path", sample_rate, length)
    elif variant == "target":
        wav = resolve_audio(ds, row, "target_path", sample_rate, length)
    elif variant == "zero":
        wav = torch.zeros(length)
    elif variant == "short1s":
        wav = resolve_audio(ds, row, "enrollment_path", sample_rate, length)
        wav[int(sample_rate) :] = 0
    elif variant == "short2s":
        wav = resolve_audio(ds, row, "enrollment_path", sample_rate, length)
        wav[int(2 * sample_rate) :] = 0
    elif variant == "noise10":
        wav = resolve_audio(ds, row, "enrollment_path", sample_rate, length)
        wav = add_noise(wav, 10.0, generator)
    elif variant == "noise5":
        wav = resolve_audio(ds, row, "enrollment_path", sample_rate, length)
        wav = add_noise(wav, 5.0, generator)
    else:
        raise ValueError(f"unknown enrollment variant: {variant}")
    return wav * norm_gain.cpu()


def add_noise(wav: torch.Tensor, snr_db: float, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(wav.shape, generator=generator)
    wav_power = wav.pow(2).mean().clamp_min(1e-8)
    noise_power = noise.pow(2).mean().clamp_min(1e-8)
    scale = (wav_power / (10 ** (snr_db / 10.0)) / noise_power).sqrt()
    return wav + scale * noise


def evaluate_variant(model, ds: TSEDataset, args: argparse.Namespace, device: torch.device, enrollment_variant: str, egsp_strength: float) -> dict[str, float]:
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=tse_collate)
    rows = []
    model.eval()
    generator = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            row = ds.rows[i]
            shuffled_row = ds.rows[(i + 17) % len(ds.rows)]
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            norm_gain = batch["norm_gain"][0]
            enrollment = make_enrollment(
                ds,
                row,
                enrollment_variant,
                args.sample_rate,
                mixture.shape[-1],
                norm_gain,
                shuffled_row,
                generator,
            ).unsqueeze(0).to(device)
            mixture_in = egsp_waveform(
                mixture,
                enrollment,
                egsp_strength,
                args.egsp_min_weight,
                args.egsp_max_weight,
                args.egsp_n_fft,
                args.egsp_hop_length,
            )
            est = model(mixture_in, enrollment)
            if est.dim() == 3:
                est = est.squeeze(1)
            interferer = resolve_audio(ds, row, "interferer_path", args.sample_rate, mixture.shape[-1])
            interferer = interferer * norm_gain.cpu()
            rows.append(
                compute_bss_metrics(
                    est[0].cpu(),
                    target[0].cpu(),
                    interferer.cpu(),
                    mixture[0].cpu(),
                    args.sample_rate,
                )
            )

    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
        return sum(vals) / len(vals) if vals else float("nan")

    return {key: mean(key) for key in ["si_sdr", "si_sdri", "sdr", "sir", "sar", "stoi", "pesq"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["egsp", "enrollment"], required=True)
    parser.add_argument("--egsp_strengths", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--enrollment_variants", default="correct,shuffled,interferer,target,zero,short1s,short2s,noise10,noise5")
    parser.add_argument("--egsp_min_weight", type=float, default=0.8)
    parser.add_argument("--egsp_max_weight", type=float, default=1.2)
    parser.add_argument("--egsp_n_fft", type=int, default=512)
    parser.add_argument("--egsp_hop_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = argparse.Namespace(**ckpt["args"])
    model_args.device = str(device)
    model = build_model(model_args).to(device)
    model.load_state_dict(ckpt["model"])
    ds = TSEDataset(args.test_manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "egsp":
        jobs = [("correct", float(item)) for item in args.egsp_strengths.split(",") if item]
    else:
        jobs = [(item, 0.0) for item in args.enrollment_variants.split(",") if item]

    rows = []
    for enrollment_variant, egsp_strength in jobs:
        summary = evaluate_variant(model, ds, args, device, enrollment_variant, egsp_strength)
        row = {"enrollment_variant": enrollment_variant, "egsp_strength": egsp_strength, **summary}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    fieldnames = ["enrollment_variant", "egsp_strength", "si_sdr", "si_sdri", "sdr", "sir", "sar", "stoi", "pesq"]
    with (out_dir / f"{args.mode}_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"{args.mode}_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
