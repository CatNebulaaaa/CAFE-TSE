from __future__ import annotations

import argparse
import csv
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cafe_tse.datasets.collate import tse_collate
from cafe_tse.datasets.tse_dataset import TSEDataset
from cafe_tse.engine.checkpoint import load_checkpoint
from cafe_tse.metrics.efficiency import count_params, skip_ratio
from cafe_tse.models.cafe_tse import build_model
from cafe_tse.utils.config import apply_overrides, load_config, select_device


def _try_thop(model: torch.nn.Module, mixture: torch.Tensor, enrollment: torch.Tensor) -> tuple[float | None, str]:
    try:
        from thop import profile

        class WavOnly(torch.nn.Module):
            def __init__(self, wrapped: torch.nn.Module):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, mix: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
                return self.wrapped(mix, enroll).wav

        profile_model = copy.deepcopy(model).eval()
        macs, _ = profile(WavOnly(profile_model), inputs=(mixture, enrollment), verbose=False)
        return float(macs), ""
    except Exception as exc:
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    device = torch.device(select_device(args.device or cfg.get("device", "cuda")))
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=str(device))
    model.eval()

    ds = TSEDataset(
        args.test_manifest,
        int(cfg["sample_rate"]),
        float(cfg.get("segment_seconds", 4.0)),
        bool(cfg.get("data", {}).get("normalize_audio", True)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=tse_collate)

    rows = []
    params = count_params(model)
    peak_memory_mb = 0.0
    macs, macs_warning = None, ""
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.num_samples + args.warmup:
                break
            mixture = batch["mixture"].to(device)
            enrollment = batch["enrollment"].to(device)
            if idx == 0:
                macs, macs_warning = _try_thop(model, mixture, enrollment)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            out = model(mixture, enrollment)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory_mb = max(peak_memory_mb, torch.cuda.max_memory_allocated(device) / (1024**2))
            elapsed = time.perf_counter() - start
            if idx < args.warmup:
                continue
            audio_dur = mixture.shape[-1] / float(cfg["sample_rate"])
            active = float(out.active_blocks[0])
            full = float(model.full_blocks)
            rows.append(
                {
                    "utt_id": batch["utt_id"][0],
                    "rtf_wall": elapsed / max(audio_dur, 1e-8),
                    "rtf_model": out.rtf if out.rtf is not None else elapsed / max(audio_dur, 1e-8),
                    "active_blocks": active,
                    "skip_ratio": skip_ratio(out.active_blocks, model.full_blocks),
                    "params": params,
                    "macs_thop": macs if macs is not None else "",
                    "active_macs_proxy": (macs * active / full) if macs is not None and full else "",
                    "peak_memory_mb": peak_memory_mb,
                    "route": out.route[0],
                }
            )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else ["utt_id"]
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
        return sum(vals) / max(len(vals), 1)

    summary = {
        "num_samples": len(rows),
        "params": params,
        "rtf_wall": mean("rtf_wall"),
        "rtf_model": mean("rtf_model"),
        "active_blocks": mean("active_blocks"),
        "skip_ratio": mean("skip_ratio"),
        "macs_thop": macs,
        "active_macs_proxy": mean("active_macs_proxy") if macs is not None else None,
        "peak_memory_mb": peak_memory_mb,
        "macs_warning": macs_warning,
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
