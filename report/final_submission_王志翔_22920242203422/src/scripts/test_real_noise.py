"""Real-environment noise robustness test (improved).

Two-phase evaluation:
  Phase 1: Run clean inference on N samples, filter to keep only samples
           with clean SI-SDR > threshold.
  Phase 2: On filtered samples, add DEMAND noise at +20, +10, +5 dB SNR
           and run inference.

Reports mean, median, min, max for each condition.
Saves per-sample metrics CSV and summary JSON.

Usage on cloud server:
    cd /root/CAFE-TSE
    PYTHONPATH=src:third_party/speakerbeam/src:third_party/asteroid_site:scripts \\
    python scripts/test_real_noise.py \\
        --checkpoint experiments/open_speakerbeam_shared_clean80_student_mid_distill_ft_continue_w005/best.pt \\
        --test_manifest data/metadata/librispeech_tse_shared_clean80/test_manifest_final.csv \\
        --noise_wav experiments/additional_challenge/prepared_wavs/noise_cafeteria.wav \\
        --out_dir experiments/additional_challenge/real_noise \\
        --n_samples 50 --min_clean_sisdr 0.0
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "speakerbeam" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "asteroid_site"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_open_speakerbeam import build_model  # noqa: E402
from cafe_tse.datasets.tse_dataset import TSEDataset  # noqa: E402
from cafe_tse.datasets.collate import tse_collate  # noqa: E402
from cafe_tse.metrics.separation import compute_bss_metrics  # noqa: E402
from cafe_tse.utils.audio_io import read_wav, fix_length, write_wav  # noqa: E402


def add_noise(signal: torch.Tensor, noise: torch.Tensor, snr_db: float,
              noise_offset: int = 0) -> torch.Tensor:
    sig_power = signal.pow(2).mean().clamp_min(1e-10)
    offset = noise_offset % max(noise.numel() - signal.numel(), 1)
    n = noise[offset:offset + signal.numel()]
    if n.numel() < signal.numel():
        repeats = signal.numel() // n.numel() + 1
        n = noise[:signal.numel()] if noise.numel() >= signal.numel() else n.repeat(repeats)[:signal.numel()]
    noise_power = n.pow(2).mean().clamp_min(1e-10)
    scale = (sig_power / (10 ** (snr_db / 10.0)) / noise_power).sqrt()
    return signal + scale * n


def resolve_audio(ds: TSEDataset, row: dict, key: str,
                  sample_rate: int, length: int) -> torch.Tensor:
    wav, _ = read_wav(ds._resolve(str(row[key])), target_sr=sample_rate)
    return fix_length(wav, length)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--noise_wav", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--min_clean_sisdr", type=float, default=0.0,
                        help="Filter: keep only samples with clean SI-SDR > this threshold")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_audio", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = argparse.Namespace(**ckpt["args"])
    model_args.device = str(device)
    model = build_model(model_args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sr = int(model_args.sample_rate)
    seg = int(sr * float(model_args.segment_seconds))

    ds = TSEDataset(args.test_manifest, sr, float(model_args.segment_seconds),
                    normalize_audio=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=tse_collate)

    if args.noise_wav and Path(args.noise_wav).exists():
        noise_raw, _ = read_wav(args.noise_wav, target_sr=sr)
        noise_type = "DEMAND"
    else:
        noise_raw = torch.randn(sr * 120)
        noise_type = "gaussian"
    print(f"Noise: {noise_type}, duration: {noise_raw.numel() / sr:.1f}s")
    print(f"Phase 1: testing {args.n_samples} samples clean, "
          f"filter threshold SI-SDR > {args.min_clean_sisdr} dB")

    # === Phase 1: Clean pass with filtering ===
    clean_results: list[dict] = []
    for i, batch in enumerate(loader):
        if i >= args.n_samples:
            break
        row = ds.rows[i]
        utt_id = str(batch["utt_id"][0])
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        norm_gain = batch["norm_gain"][0]
        enrollment = resolve_audio(ds, row, "enrollment_path", sr, seg)
        enrollment = (enrollment * norm_gain.cpu()).unsqueeze(0).to(device)
        enr_dur = float(row.get("enrollment_length", 4.0))
        difficulty = str(row.get("difficulty", "unknown"))

        with torch.no_grad():
            t0 = time.perf_counter()
            est = model(mixture, enrollment)
            t1 = time.perf_counter()
        if est.dim() == 3:
            est = est.squeeze(1)

        interferer = resolve_audio(ds, row, "interferer_path", sr, seg)
        interferer = interferer * norm_gain.cpu()
        metrics = compute_bss_metrics(est[0].cpu(), target[0].cpu(),
                                      interferer, mixture[0].cpu(), sr)
        metrics["utt_id"] = utt_id
        metrics["difficulty"] = difficulty
        metrics["enrollment_dur"] = enr_dur
        metrics["rtf"] = round((t1 - t0) / float(seg / sr), 5)
        clean_results.append(metrics)

    # Filter
    kept = [r for r in clean_results
            if r["si_sdr"] > args.min_clean_sisdr]
    skipped = len(clean_results) - len(kept)
    print(f"Phase 1 done: {len(clean_results)} tested, {len(kept)} kept "
          f"(filter: clean SI-SDR > {args.min_clean_sisdr} dB, skipped {skipped})")

    if not kept:
        print("ERROR: no samples passed filter. Lower --min_clean_sisdr.")
        return

    # Show clean stats on kept samples
    kept_sisdr = [r["si_sdr"] for r in kept]
    print(f"Kept clean SI-SDR: mean={statistics.mean(kept_sisdr):.2f}, "
          f"median={statistics.median(kept_sisdr):.2f}, "
          f"min={min(kept_sisdr):.2f}, max={max(kept_sisdr):.2f}")

    # === Phase 2: Noise conditions on kept samples ===
    conditions = [
        ("clean", None),
        ("noise_20dB", 20.0),
        ("noise_10dB", 10.0),
        ("noise_5dB", 5.0),
    ]

    # Phase 2: re-process kept samples with noise conditions
    rows: list[dict] = []

    # Re-iterate loader and only process kept indices
    kept_set = set(clean_results.index(r) for r in kept)
    for i, batch in enumerate(loader):
        if i >= args.n_samples:
            break
        if i not in kept_set:
            continue

        row = ds.rows[i]
        utt_id = str(batch["utt_id"][0])
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        norm_gain = batch["norm_gain"][0]
        enrollment = resolve_audio(ds, row, "enrollment_path", sr, seg)
        enrollment = (enrollment * norm_gain.cpu()).unsqueeze(0).to(device)

        for cond_name, snr_db in conditions:
            if snr_db is None:
                mix_in = mixture
            else:
                noise_offset = i * seg
                noise_added = add_noise(mixture[0].cpu(), noise_raw, snr_db,
                                        noise_offset)
                mix_in = noise_added.unsqueeze(0).to(device)

            with torch.no_grad():
                t0 = time.perf_counter()
                est = model(mix_in, enrollment)
                t1 = time.perf_counter()
            if est.dim() == 3:
                est = est.squeeze(1)

            interferer = resolve_audio(ds, row, "interferer_path", sr, seg)
            interferer = interferer * norm_gain.cpu()
            metrics = compute_bss_metrics(est[0].cpu(), target[0].cpu(),
                                          interferer, mix_in[0].cpu(), sr)
            row_data = {
                "utt_id": utt_id,
                "condition": cond_name,
                "snr_db": snr_db if snr_db is not None else "clean",
                "noise_type": noise_type,
                **{k: round(float(v), 4) for k, v in metrics.items()
                   if v == v},  # filter NaN
                "rtf": round((t1 - t0) / float(seg / sr), 5),
            }
            rows.append(row_data)

            if args.save_audio and cond_name != "clean":
                write_wav(audio_dir / f"{utt_id}_{cond_name}_estimated.wav",
                          est[0].cpu(), sr)

    # === Save ===
    fieldnames = ["utt_id", "condition", "snr_db", "noise_type",
                  "si_sdr", "si_sdri", "sdr", "sir", "sar", "stoi", "pesq", "rtf"]
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        # Filter rows to only include existing fields
        clean_rows = [{k: v for k, v in r.items() if k in fieldnames} for r in rows]
        writer.writerows(clean_rows)
    print(f"Metrics saved: {csv_path} ({len(clean_rows)} rows)")

    # Summary
    summary = {
        "filter": f"clean SI-SDR > {args.min_clean_sisdr} dB",
        "n_tested": len(clean_results),
        "n_kept": len(kept),
        "n_skipped": skipped,
        "noise_type": noise_type,
        "conditions": {},
    }
    for cond_name, _ in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond_name]
        if not cond_rows:
            continue
        vals_sisdr = [r["si_sdr"] for r in cond_rows if r.get("si_sdr") == r.get("si_sdr")]
        vals_sisdri = [r["si_sdri"] for r in cond_rows if r.get("si_sdri") == r.get("si_sdri")]
        vals_sdr = [r["sdr"] for r in cond_rows if r.get("sdr") == r.get("sdr")]
        summary["conditions"][cond_name] = {
            "n": len(cond_rows),
            "si_sdr_mean": round(statistics.mean(vals_sisdr), 4),
            "si_sdr_median": round(statistics.median(vals_sisdr), 4),
            "si_sdr_min": round(min(vals_sisdr), 4),
            "si_sdr_max": round(max(vals_sisdr), 4),
            "si_sdri_mean": round(statistics.mean(vals_sisdri), 4),
            "si_sdri_median": round(statistics.median(vals_sisdri), 4),
            "sdr_mean": round(statistics.mean(vals_sdr), 4),
            "sdr_median": round(statistics.median(vals_sdr), 4),
        }
        print(f"  {cond_name}: n={len(cond_rows)}, "
              f"SI-SDR mean={statistics.mean(vals_sisdr):.2f}, "
              f"median={statistics.median(vals_sisdr):.2f}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")


if __name__ == "__main__":
    main()
