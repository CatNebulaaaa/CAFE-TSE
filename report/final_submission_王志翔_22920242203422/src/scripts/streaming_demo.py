"""Streaming target-speaker extraction demo.

Simulates real-time processing by feeding mixture wav in chunks
(with overlap-add) through a pre-trained TimeDomainSpeakerBeam model.
Saves the streamed output and a timing log.

Usage on cloud server:
    cd /root/CAFE-TSE
    python scripts/streaming_demo.py \
        --checkpoint experiments/open_speakerbeam_shared_clean80_student_mid_distill_ft_continue_w005/best.pt \
        --mixture data/librispeech_tse/test/mixture/test_00000.wav \
        --enrollment data/librispeech_tse/test/enrollment/test_00000.wav \
        --out_dir experiments/additional_challenge/streaming_demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "speakerbeam" / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "asteroid_site"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_open_speakerbeam import build_model  # noqa: E402
from cafe_tse.utils.audio_io import read_wav, write_wav  # noqa: E402
from cafe_tse.losses.sisdr import si_sdr  # noqa: E402


def rms_normalize(wav: torch.Tensor, target_rms: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(eps)
    return wav * (target_rms / rms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--enrollment", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--chunk_seconds", type=float, default=2.0)
    parser.add_argument("--hop_seconds", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", default=None, help="optional clean target wav for reference SI-SDR")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = argparse.Namespace(**ckpt["args"])
    model = build_model(model_args).to(device)
    model.eval()

    sr = int(model_args.sample_rate)
    chunk_samples = int(sr * args.chunk_seconds)
    hop_samples = int(sr * args.hop_seconds)

    mixture, _ = read_wav(args.mixture, target_sr=sr)
    enrollment, _ = read_wav(args.enrollment, target_sr=sr)

    # Pad mixture to at least one chunk
    if mixture.numel() < chunk_samples:
        mixture = torch.nn.functional.pad(mixture, (0, chunk_samples - mixture.numel()))

    enrollment_rms = rms_normalize(enrollment.clone()).unsqueeze(0).to(device)
    total_samples = mixture.numel()
    n_hops = max(1, (total_samples - chunk_samples) // hop_samples + 1)

    window = torch.hann_window(chunk_samples)  # CPU: overlap-add on CPU
    output = torch.zeros(total_samples)
    weight_sum = torch.zeros(total_samples)

    log_entries: list[dict] = []
    total_start = time.perf_counter()

    print(f"Streaming: {total_samples / sr:.1f}s audio, {n_hops} chunks "
          f"(chunk={args.chunk_seconds}s, hop={args.hop_seconds}s)")

    for i in range(n_hops):
        start_idx = i * hop_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        chunk = mixture[start_idx:end_idx].clone()

        # Pad last partial chunk
        if chunk.numel() < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.numel()))

        # RMS normalize (same as TSEDataset)
        mix_chunk = rms_normalize(chunk).unsqueeze(0).to(device)

        with torch.no_grad():
            t0 = time.perf_counter()
            est = model(mix_chunk, enrollment_rms)
            t1 = time.perf_counter()

        if est.dim() == 3:
            est = est.squeeze(1)
        est_chunk = est[0, :chunk_samples].cpu()

        # Overlap-add with Hann window
        actual_len = end_idx - start_idx
        win = window[:actual_len] if actual_len < chunk_samples else window
        output[start_idx:end_idx] += est_chunk[:actual_len] * win
        weight_sum[start_idx:end_idx] += win

        chunk_latency = t1 - t0
        chunk_rtf = chunk_latency / max(args.chunk_seconds, 1e-8)
        log_entries.append({
            "chunk": i,
            "start_sample": int(start_idx),
            "end_sample": int(end_idx),
            "latency_s": round(chunk_latency, 5),
            "rtf": round(chunk_rtf, 5),
        })

        if (i + 1) % 10 == 0 or i == n_hops - 1:
            print(f"  chunk {i + 1}/{n_hops}: latency={chunk_latency:.4f}s, RTF={chunk_rtf:.4f}")

    total_elapsed = time.perf_counter() - total_start

    # Normalize overlap-add and denormalize (inverse RMS)
    mask = weight_sum > 0
    output[mask] /= weight_sum[mask]

    # Measure output RMS and match to original mixture RMS
    mix_rms = mixture.pow(2).mean().sqrt().clamp_min(1e-8)
    out_rms = output.pow(2).mean().sqrt().clamp_min(1e-8)
    output = output * (mix_rms / out_rms)

    out_wav_path = out_dir / "output_streamed.wav"
    write_wav(out_wav_path, output.cpu(), sr)
    print(f"Saved: {out_wav_path}")

    # Compute SI-SDR if target provided
    si_sdr_val = None
    si_sdri_val = None
    if args.target:
        target, _ = read_wav(args.target, target_sr=sr)
        tlen = min(target.numel(), output.numel())
        target = target[:tlen].unsqueeze(0)
        mix_ref = mixture[:tlen].unsqueeze(0)
        out_ref = output[:tlen].unsqueeze(0)
        si_sdr_val = float(si_sdr(out_ref, target).item())
        si_sdri_val = float(si_sdr_val - si_sdr(mix_ref, target).item())
        print(f"SI-SDR: {si_sdr_val:.2f} dB, SI-SDRi: {si_sdri_val:.2f} dB")

    # Save log
    chunk_stats = {
        "n_chunks": len(log_entries),
        "mean_latency_s": round(
            sum(e["latency_s"] for e in log_entries) / len(log_entries), 5),
        "mean_rtf": round(
            sum(e["rtf"] for e in log_entries) / len(log_entries), 5),
        "total_elapsed_s": round(total_elapsed, 3),
        "audio_duration_s": round(total_samples / sr, 3),
        "overall_rtf": round(total_elapsed / (total_samples / sr), 5),
    }
    summary = {
        "mixture": args.mixture,
        "enrollment": args.enrollment,
        "chunk_seconds": args.chunk_seconds,
        "hop_seconds": args.hop_seconds,
        "sample_rate": sr,
        "si_sdr": round(si_sdr_val, 4) if si_sdr_val is not None else None,
        "si_sdri": round(si_sdri_val, 4) if si_sdri_val is not None else None,
        **chunk_stats,
        "chunk_log": log_entries,
    }
    log_path = out_dir / "log.txt"
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Log saved: {log_path}")
    print(f"Overall RTF: {chunk_stats['overall_rtf']:.4f}, "
          f"audio: {chunk_stats['audio_duration_s']:.1f}s, "
          f"total: {chunk_stats['total_elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
