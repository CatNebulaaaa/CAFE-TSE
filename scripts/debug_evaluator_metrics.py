from __future__ import annotations

import torch
import numpy as np
from pystoi import stoi

from cafe_tse.datasets.tse_dataset import TSEDataset
from cafe_tse.engine.checkpoint import load_checkpoint
from cafe_tse.metrics.separation import compute_bss_metrics
from cafe_tse.models.cafe_tse import build_model
from cafe_tse.utils.audio_io import fix_length, read_wav
from cafe_tse.utils.config import apply_overrides, load_config


def main() -> None:
    cfg = load_config("configs/cafe_tse_dynamic.yaml")
    cfg = apply_overrides(
        cfg,
        [
            "device=cuda",
            "sample_rate=8000",
            "segment_seconds=4.0",
            "model.emb_dim=40",
            "model.hidden_dim=160",
            "model.n_heads=4",
            "model.n_blocks=5",
            "model.sparse_fusion_blocks=[0,2,4]",
            "model.dynamic_inference=false",
            "model.full_blocks=5",
            "model.egsp_enabled=true",
            "model.egsp_strength=0.05",
            "model.egsp_min_weight=0.80",
            "model.egsp_max_weight=1.20",
            "model.egsp_apply_to_spec=true",
        ],
    )
    device = torch.device("cuda")
    model = build_model(cfg).to(device)
    load_checkpoint("experiments/mini_exp10_distill_5block_mid/checkpoints/best.pt", model, map_location=str(device))
    model.eval()
    ds = TSEDataset("data/metadata/minilibrimix_disjoint/test_manifest_final.csv", 8000, 4.0, True)
    item = ds[0]
    with torch.no_grad():
        out = model(item["mixture"].unsqueeze(0).to(device), item["enrollment"].unsqueeze(0).to(device))
    row = ds.rows[0]
    interferer, _ = read_wav(ds._resolve(row["interferer_path"]), target_sr=8000)
    interferer = fix_length(interferer, item["mixture"].shape[-1])
    metrics = compute_bss_metrics(out.wav[0].cpu(), item["target"].cpu(), interferer.cpu(), item["mixture"].cpu(), 8000)
    print(metrics)
    est = out.wav[0].detach().cpu().float()
    target = item["target"].detach().cpu().float()
    n = min(est.numel(), target.numel())
    est_np = np.clip(np.ascontiguousarray(est[:n].numpy(), dtype=np.float64), -1.0, 1.0)
    target_np = np.clip(np.ascontiguousarray(target[:n].numpy(), dtype=np.float64), -1.0, 1.0)
    print("ranges", float(est.min()), float(est.max()), float(target.min()), float(target.max()), "len", n)
    print("direct_stoi", stoi(target_np, est_np, 8000, extended=False))


if __name__ == "__main__":
    main()
