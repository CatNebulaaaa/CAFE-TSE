"""Chinese-English cross-lingual target speaker extraction test.

Supports two test directions:
  A. Chinese target + English interferer (zh target, zh enrollment, en interferer)
  B. English target + Chinese interferer (en target, en enrollment, zh interferer)

Each direction requires 3 wav files. The script constructs a mixture
(target + interferer at equal level), runs inference, and records metrics.

Usage on cloud server:
    cd /root/CAFE-TSE
    python scripts/test_cross_lingual.py \
        --checkpoint experiments/open_speakerbeam_shared_clean80_student_mid_distill_ft_continue_w005/best.pt \
        --zh_target /path/to/zh_target.wav \
        --zh_enroll /path/to/zh_enroll.wav \
        --en_interferer /path/to/en_interferer.wav \
        --en_target /path/to/en_target.wav \
        --en_enroll /path/to/en_enroll.wav \
        --zh_interferer /path/to/zh_interferer.wav \
        --out_dir experiments/additional_challenge/cross_lingual
"""

from __future__ import annotations

import argparse
import csv
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
from cafe_tse.losses.sisdr import si_sdr  # noqa: E402
from cafe_tse.utils.audio_io import read_wav, fix_length, write_wav  # noqa: E402


def rms_normalize(wav: torch.Tensor, target_rms: float = 0.05, eps: float = 1e-8) -> torch.Tensor:
    rms = wav.pow(2).mean().sqrt().clamp_min(eps)
    return wav * (target_rms / rms)


def mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float = 0.0) -> torch.Tensor:
    """Mix two signals at given SNR (signal is reference)."""
    sig_pow = signal.pow(2).mean().clamp_min(1e-10)
    noi_pow = noise.pow(2).mean().clamp_min(1e-10)
    scale = (sig_pow / (10 ** (snr_db / 10.0)) / noi_pow).sqrt()
    return signal + scale * noise


def run_case(model, mixture: torch.Tensor, enrollment: torch.Tensor,
             target: torch.Tensor, device: torch.device, seg: int) -> dict:
    """Run one inference case and return metrics."""
    mix = fix_length(mixture, seg)
    enroll = fix_length(enrollment, seg)
    tgt = fix_length(target, seg)

    # Normalize both to RMS ~0.05 (matching training distribution)
    mix_rms = mix.pow(2).mean().sqrt().clamp_min(1e-8)
    norm_gain = 0.05 / mix_rms
    mix_norm = (mix * norm_gain).unsqueeze(0).to(device)
    # Use same norm_gain for enrollment (matching evaluate_open_speakerbeam_variants)
    enroll_norm = (enroll * norm_gain).unsqueeze(0).to(device)
    tgt_norm = tgt * norm_gain

    with torch.no_grad():
        t0 = time.perf_counter()
        est = model(mix_norm, enroll_norm)
        t1 = time.perf_counter()
    if est.dim() == 3:
        est = est.squeeze(1)

    # Compute SI-SDR on normalized scale (scale-invariant)
    est_out = est[0].cpu()
    si_sdr_val = float(si_sdr(est_out.unsqueeze(0), tgt_norm.unsqueeze(0)).item())
    mix_score = float(si_sdr(mix_norm[0].cpu().unsqueeze(0), tgt_norm.unsqueeze(0)).item())

    return {
        "si_sdr": round(si_sdr_val, 4),
        "si_sdri": round(si_sdr_val - mix_score, 4),
        "rtf": round((t1 - t0) / (seg / 8000), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--zh_target", default=None)
    parser.add_argument("--zh_enroll", default=None)
    parser.add_argument("--en_interferer", default=None)
    parser.add_argument("--en_target", default=None)
    parser.add_argument("--en_enroll", default=None)
    parser.add_argument("--zh_interferer", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_args = argparse.Namespace(**ckpt["args"])
    model = build_model(model_args).to(device)
    model.eval()
    sr = int(model_args.sample_rate)
    seg = int(sr * float(model_args.segment_seconds))

    rows: list[dict] = []
    missing = []

    # --- Case A: Chinese target + English interferer ---
    if args.zh_target and args.zh_enroll and args.en_interferer:
        zh_tgt, _ = read_wav(args.zh_target, target_sr=sr)
        zh_enr, _ = read_wav(args.zh_enroll, target_sr=sr)
        en_int, _ = read_wav(args.en_interferer, target_sr=sr)
        mixture_a = mix_at_snr(zh_tgt, en_int, snr_db=0.0)

        result_a = run_case(model, mixture_a, zh_enr, zh_tgt, device, seg)
        result_a["case"] = "zh_target_en_interferer"
        result_a["target_lang"] = "zh"
        result_a["interferer_lang"] = "en"
        rows.append(result_a)

        write_wav(audio_dir / "case_a_zh_target_en_interferer_mixture.wav", mixture_a, sr)
        write_wav(audio_dir / "case_a_estimated.wav",
                  fix_length(read_wav(args.zh_target, target_sr=sr)[0], seg), sr)
        print(f"Case A (zh→en): SI-SDR={result_a['si_sdr']:.2f}, SI-SDRi={result_a['si_sdri']:.2f}")
    else:
        missing.append("case_a (needs --zh_target, --zh_enroll, --en_interferer)")

    # --- Case B: English target + Chinese interferer ---
    if args.en_target and args.en_enroll and args.zh_interferer:
        en_tgt, _ = read_wav(args.en_target, target_sr=sr)
        en_enr, _ = read_wav(args.en_enroll, target_sr=sr)
        zh_int, _ = read_wav(args.zh_interferer, target_sr=sr)
        mixture_b = mix_at_snr(en_tgt, zh_int, snr_db=0.0)

        result_b = run_case(model, mixture_b, en_enr, en_tgt, device, seg)
        result_b["case"] = "en_target_zh_interferer"
        result_b["target_lang"] = "en"
        result_b["interferer_lang"] = "zh"
        rows.append(result_b)

        write_wav(audio_dir / "case_b_en_target_zh_interferer_mixture.wav", mixture_b, sr)
        write_wav(audio_dir / "case_b_estimated.wav",
                  fix_length(read_wav(args.en_target, target_sr=sr)[0], seg), sr)
        print(f"Case B (en→zh): SI-SDR={result_b['si_sdr']:.2f}, SI-SDRi={result_b['si_sdri']:.2f}")
    else:
        missing.append("case_b (needs --en_target, --en_enroll, --zh_interferer)")

    # --- Baseline: matched-language reference using existing test data ---
    print("\nMatched-language baselines (English-English from test set):")
    # Load 3 easy test samples for reference
    from cafe_tse.datasets.tse_dataset import TSEDataset  # noqa: E402
    from cafe_tse.datasets.collate import tse_collate  # noqa: E402
    import os
    manifest_path = os.path.join(os.path.dirname(args.checkpoint), "..", "..", "..",
                                 "data", "metadata", "librispeech_tse", "test_manifest_final.csv")
    if os.path.exists(manifest_path):
        ds = TSEDataset(manifest_path, sr, float(model_args.segment_seconds), normalize_audio=True)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=tse_collate)
        for i, batch_ in enumerate(loader):
            if i >= 3:
                break
            mix = batch_["mixture"].to(device)
            enr_ds = batch_["enrollment"].to(device)
            tgt = batch_["target"].to(device)
            ng = batch_["norm_gain"][0]
            row = ds.rows[i]
            # Use same enrollment approach as working eval
            enr_raw = fix_length(read_wav(ds._resolve(str(row["enrollment_path"])), target_sr=sr)[0], seg)
            enr = (enr_raw * ng.cpu()).unsqueeze(0).to(device)

            with torch.no_grad():
                est = model(mix, enr)
            if est.dim() == 3:
                est = est.squeeze(1)
            s = float(si_sdr(est, tgt).item())
            si = float(s - si_sdr(mix, tgt).item())
            rows.append({
                "case": f"en_en_baseline_{i}",
                "target_lang": "en",
                "interferer_lang": "en",
                "si_sdr": round(s, 4),
                "si_sdri": round(si, 4),
                "rtf": None,
            })
            print(f"  en-en baseline {i}: SI-SDR={s:.2f}, SI-SDRi={si:.2f}")

    # Save
    csv_path = out_dir / "metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nMetrics saved: {csv_path}")

    summary = {"n_cases": len(rows), "missing": missing, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if missing:
        print(f"\nMissing inputs: {missing}")
        print("Provide wav files for cross-lingual cases to complete the experiment.")


if __name__ == "__main__":
    main()
