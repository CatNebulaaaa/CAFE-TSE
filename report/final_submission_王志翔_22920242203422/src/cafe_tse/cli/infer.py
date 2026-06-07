from __future__ import annotations

import argparse
import time

import torch

from cafe_tse.engine.checkpoint import load_checkpoint
from cafe_tse.models.cafe_tse import build_model
from cafe_tse.utils.audio_io import fix_length, read_wav, write_wav
from cafe_tse.utils.config import apply_overrides, load_config, select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--enrollment", required=True)
    parser.add_argument("--out_wav", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    device = torch.device(select_device(args.device))
    sr = int(cfg["sample_rate"])
    segment = int(sr * float(cfg.get("segment_seconds", 4.0)))
    mixture, _ = read_wav(args.mixture, target_sr=sr)
    enrollment, _ = read_wav(args.enrollment, target_sr=sr)
    mixture = fix_length(mixture, segment).unsqueeze(0).to(device)
    enrollment = fix_length(enrollment, segment).unsqueeze(0).to(device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        out = model(mixture, enrollment)
        elapsed = time.perf_counter() - start
    write_wav(args.out_wav, out.wav[0].cpu(), sr)
    rtf = elapsed / max(mixture.shape[-1] / sr, 1e-8)
    print(f"complexity_score: {float(out.complexity_score[0].cpu()):.4f}")
    print(f"route: {out.route[0]}")
    print(f"active_blocks: {out.active_blocks[0]}")
    print(f"rtf: {rtf:.4f}")
    print(f"saved: {args.out_wav}")


if __name__ == "__main__":
    main()
