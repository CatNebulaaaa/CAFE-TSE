from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch

from cafe_tse.losses.sisdr import si_sdr
from cafe_tse.metrics.separation import compute_bss_metrics
from cafe_tse.utils.audio_io import fix_length, read_wav


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    return (manifest.parent / path).resolve()


def _rms(wav: torch.Tensor) -> float:
    return float(wav.float().pow(2).mean().sqrt())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--max_rows", type=int, default=300)
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    rows = rows[: args.max_rows] if args.max_rows else rows
    seg = int(args.sample_rate * args.segment_seconds)

    missing = Counter()
    speakers = Counter()
    same_target_enroll = 0
    missing_interferer = 0
    same_basename = 0
    mix_sisdr = []
    mix_sdr = []
    target_identity_sdr = []
    recon_rel = []
    corr_abs = []
    target_rms = []
    enroll_rms = []
    nonzero_ratio = []

    for row in rows:
        speakers[row.get("speaker_id", "")] += 1
        target_path = _resolve(manifest, row["target_path"])
        enroll_path = _resolve(manifest, row["enrollment_path"])
        mixture_path = _resolve(manifest, row["mixture_path"])
        interferer_value = row.get("interferer_path", "")
        interferer_path = _resolve(manifest, interferer_value) if interferer_value else None
        for label, path in [("mixture", mixture_path), ("target", target_path), ("enrollment", enroll_path)]:
            if not path.exists():
                missing[label] += 1
        if interferer_path is None or not interferer_path.exists():
            missing_interferer += 1
        if str(target_path) == str(enroll_path):
            same_target_enroll += 1
        if target_path.name == enroll_path.name:
            same_basename += 1
        if not mixture_path.exists() or not target_path.exists() or not enroll_path.exists():
            continue

        mixture, _ = read_wav(mixture_path, args.sample_rate)
        target, _ = read_wav(target_path, args.sample_rate)
        enrollment, _ = read_wav(enroll_path, args.sample_rate)
        mixture = fix_length(mixture, seg)
        target = fix_length(target, seg)
        enrollment = fix_length(enrollment, seg)
        target_rms.append(_rms(target))
        enroll_rms.append(_rms(enrollment))
        nonzero_ratio.append(float((target.abs() > 1e-5).float().mean()))
        mix_sisdr.append(float(si_sdr(mixture[None], target[None])))
        centered_t = target - target.mean()
        centered_e = enrollment - enrollment.mean()
        denom = centered_t.norm() * centered_e.norm()
        corr_abs.append(float((centered_t @ centered_e).abs() / denom.clamp_min(1e-8)))
        if interferer_path is not None and interferer_path.exists():
            interferer, _ = read_wav(interferer_path, args.sample_rate)
            interferer = fix_length(interferer, seg)
            bss = compute_bss_metrics(mixture, target, interferer, mixture, args.sample_rate)
            mix_sdr.append(bss["sdr"])
            target_identity_sdr.append(compute_bss_metrics(target, target, interferer, mixture, args.sample_rate)["sdr"])
            recon_rel.append(_rms(mixture - target - interferer) / max(_rms(mixture), 1e-8))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    print(f"manifest={manifest}")
    print(f"rows_checked={len(rows)} speakers={len(speakers)} speaker_count_min={min(speakers.values()) if speakers else 0} speaker_count_max={max(speakers.values()) if speakers else 0}")
    print(f"missing={dict(missing)} missing_interferer={missing_interferer}")
    print(f"target_eq_enrollment={same_target_enroll} target_enrollment_same_basename={same_basename}")
    print(f"mix_si_sdr_mean={mean(mix_sisdr):.4f} mix_sdr_mean={mean(mix_sdr):.4f}")
    print(f"target_identity_sdr_mean={mean(target_identity_sdr):.4f} mixture_minus_sources_rel_rms={mean(recon_rel):.6f}")
    print(f"abs_corr_enrollment_target_mean={mean(corr_abs):.4f}")
    print(f"target_rms_mean={mean(target_rms):.6f} enrollment_rms_mean={mean(enroll_rms):.6f} target_nonzero_ratio_mean={mean(nonzero_ratio):.4f}")


if __name__ == "__main__":
    main()
