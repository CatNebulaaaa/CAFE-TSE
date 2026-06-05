from __future__ import annotations

import json
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cafe_tse.datasets.collate import tse_collate
from cafe_tse.datasets.tse_dataset import TSEDataset
from cafe_tse.engine.checkpoint import load_checkpoint
from cafe_tse.metrics.efficiency import count_params, skip_ratio
from cafe_tse.metrics.separation import compute_basic_metrics, compute_bss_metrics
from cafe_tse.models.cafe_tse import build_model
from cafe_tse.utils.audio_io import fix_length, read_wav, rms_normalize, write_wav
from cafe_tse.utils.config import select_device


class Evaluator:
    def __init__(self, cfg: dict, checkpoint: str, test_manifest: str, out_dir: str, device: str | None = None):
        self.cfg = cfg
        self.device = torch.device(select_device(device or cfg.get("device", "cuda")))
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = build_model(cfg).to(self.device)
        load_checkpoint(checkpoint, self.model, map_location=str(self.device))
        self.model.eval()
        self.ds = TSEDataset(
            test_manifest,
            int(cfg["sample_rate"]),
            float(cfg.get("segment_seconds", 4.0)),
            bool(cfg.get("data", {}).get("normalize_audio", True)),
        )
        self.normalize_audio = bool(cfg.get("data", {}).get("normalize_audio", True))

    def run(self, save_audio: int = 0) -> dict:
        loader = DataLoader(self.ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=tse_collate)
        rows = []
        audio_dir = self.out_dir / "audio"
        params = count_params(self.model)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(loader, desc="evaluate")):
                mixture = batch["mixture"].to(self.device)
                target = batch["target"].to(self.device)
                enrollment = batch["enrollment"].to(self.device)
                out = self.model(mixture, enrollment)
                row_src = self.ds.rows[i]
                interferer_path = str(row_src.get("interferer_path", ""))
                if interferer_path:
                    interferer, _ = read_wav(self.ds._resolve(interferer_path), target_sr=int(self.cfg["sample_rate"]))
                    interferer = fix_length(interferer, mixture.shape[-1])
                    if self.normalize_audio:
                        interferer = interferer * batch["norm_gain"][0].cpu()
                    metrics = compute_bss_metrics(
                        out.wav[0].cpu(),
                        target[0].cpu(),
                        interferer.cpu(),
                        mixture[0].cpu(),
                        int(self.cfg["sample_rate"]),
                    )
                else:
                    metrics = compute_basic_metrics(out.wav[0].cpu(), target[0].cpu(), mixture[0].cpu(), int(self.cfg["sample_rate"]))
                row = {
                    "utt_id": batch["utt_id"][0],
                    **metrics,
                    "rtf": out.rtf if out.rtf is not None else float("nan"),
                    "route": out.route[0],
                    "active_blocks": out.active_blocks[0],
                    "complexity_score": float(out.complexity_score[0].cpu()),
                    "difficulty": batch["difficulty"][0],
                    "params": params,
                    "skip_ratio": skip_ratio(out.active_blocks, self.model.full_blocks),
                }
                rows.append(row)
                if i < save_audio:
                    write_wav(audio_dir / f"{batch['utt_id'][0]}_estimated.wav", out.wav[0].cpu(), int(self.cfg["sample_rate"]))
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (self.out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames if fieldnames else ["utt_id"])
            writer.writeheader()
            writer.writerows(rows)
        def mean(key: str) -> float:
            vals = [float(r[key]) for r in rows if r.get(key) == r.get(key)]
            if not vals:
                return float("nan")
            return sum(vals) / len(vals)
        summary = {
            "num_samples": len(rows),
            "si_sdr": mean("si_sdr") if rows else float("nan"),
            "si_sdri": mean("si_sdri") if rows else float("nan"),
            "sdr": mean("sdr") if rows else float("nan"),
            "sir": mean("sir") if rows else float("nan"),
            "sar": mean("sar") if rows else float("nan"),
            "stoi": mean("stoi") if rows else float("nan"),
            "pesq": mean("pesq") if rows else float("nan"),
            "rtf": mean("rtf") if rows else float("nan"),
            "skip_ratio": mean("skip_ratio") if rows else float("nan"),
            "params": params,
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
