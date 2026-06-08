from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "asteroid_site"))
sys.path.insert(0, str(ROOT / "third_party" / "speakerbeam" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_open_speakerbeam import build_model  # noqa: E402

from asteroid.models.base_models import _shape_reconstructed, _unsqueeze_to_3d  # noqa: E402
from asteroid.utils.torch_utils import jitable_shape, pad_x_to_y  # noqa: E402
from cafe_tse.datasets.collate import tse_collate  # noqa: E402
from cafe_tse.datasets.tse_dataset import TSEDataset  # noqa: E402
from cafe_tse.metrics.separation import compute_bss_metrics  # noqa: E402
from cafe_tse.utils.audio_io import fix_length, read_wav, rms_normalize  # noqa: E402
from cafe_tse.utils.manifest import read_manifest  # noqa: E402


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (manifest.parent / path).resolve() if not path.exists() else path


def _load_enroll(path: Path, sample_rate: int, length: int) -> torch.Tensor:
    wav, _ = read_wav(path, target_sr=sample_rate)
    return rms_normalize(fix_length(wav, length)).float()


def _mean(rows: list[dict[str, float]], key: str) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
    return sum(vals) / len(vals) if vals else float("nan")


def build_pool(pool_manifest: str, sample_rate: int, length: int, max_per_spk: int) -> dict[str, list[torch.Tensor]]:
    manifest_path = Path(pool_manifest)
    rows = read_manifest(pool_manifest)
    pool: dict[str, list[torch.Tensor]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        spk = str(row["speaker_id"])
        if len(pool[spk]) >= max_per_spk:
            continue
        path = _resolve(manifest_path, str(row["enrollment_path"]))
        key = (spk, str(path))
        if key in seen:
            continue
        seen.add(key)
        pool[spk].append(_load_enroll(path, sample_rate, length))
    return pool


def forward_with_embedding(model, wav: torch.Tensor, enroll_emb: torch.Tensor) -> torch.Tensor:
    shape = jitable_shape(wav)
    wav_3d = _unsqueeze_to_3d(wav)
    tf_rep = model.forward_encoder(wav_3d)
    est_masks = model.forward_masker(tf_rep, enroll_emb)
    masked_tf_rep = model.apply_masks(tf_rep, est_masks)
    decoded = model.forward_decoder(masked_tf_rep)
    reconstructed = pad_x_to_y(decoded, wav_3d)
    return _shape_reconstructed(reconstructed, shape)


def evaluate_k(model, test_manifest: str, pool_manifest: str, k: int, args, device) -> dict[str, float]:
    ds = TSEDataset(test_manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)
    length = int(args.sample_rate * args.segment_seconds)
    pool = build_pool(pool_manifest, args.sample_rate, length, max(k, args.max_pool_per_speaker))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=tse_collate)
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            row_src = ds.rows[i]
            spk = str(row_src["speaker_id"])
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            enrollments = [batch["enrollment"][0]]
            for wav in pool.get(spk, []):
                if len(enrollments) >= k:
                    break
                enrollments.append(wav)
            embeddings = [model.auxiliary(enroll.unsqueeze(0).to(device)) for enroll in enrollments[:k]]
            enroll_emb = torch.stack(embeddings, dim=0).mean(dim=0)
            est = forward_with_embedding(model, mixture, enroll_emb)
            if est.dim() == 3:
                est = est.squeeze(1)
            interferer, _ = read_wav(ds._resolve(str(row_src["interferer_path"])), target_sr=args.sample_rate)
            interferer = fix_length(interferer, mixture.shape[-1])
            interferer = interferer * batch["norm_gain"][0].cpu()
            rows.append(
                compute_bss_metrics(
                    est[0].cpu(),
                    target[0].cpu(),
                    interferer.cpu(),
                    mixture[0].cpu(),
                    args.sample_rate,
                )
            )
    return {key: _mean(rows, key) for key in ["si_sdr", "si_sdri", "sdr", "sir", "sar", "stoi", "pesq"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--pool_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--ks", default="1,2,4,8")
    parser.add_argument("--max_pool_per_speaker", type=int, default=8)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(argparse.Namespace(**ckpt["args"])).to(device)
    model.load_state_dict(ckpt["model"])

    rows = []
    for k in [int(x) for x in args.ks.split(",") if x.strip()]:
        summary = evaluate_k(model, args.test_manifest, args.pool_manifest, k, args, device)
        row = {"k": k, **summary}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    fieldnames = ["k", "si_sdr", "si_sdri", "sdr", "sir", "sar", "stoi", "pesq"]
    with (out_dir / "embedding_pool_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda r: float(r["si_sdr"]))
    (out_dir / "best_embedding_pool.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
