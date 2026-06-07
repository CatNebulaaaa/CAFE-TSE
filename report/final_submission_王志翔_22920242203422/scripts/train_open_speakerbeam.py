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

from models.td_speakerbeam import TimeDomainSpeakerBeam  # noqa: E402

from cafe_tse.datasets.collate import tse_collate  # noqa: E402
from cafe_tse.datasets.tse_dataset import TSEDataset  # noqa: E402
from cafe_tse.losses.sisdr import si_sdr, si_sdr_loss  # noqa: E402
from cafe_tse.metrics.separation import compute_bss_metrics  # noqa: E402
from cafe_tse.utils.audio_io import fix_length, read_wav  # noqa: E402


def build_model(args: argparse.Namespace) -> TimeDomainSpeakerBeam:
    return TimeDomainSpeakerBeam(
        i_adapt_layer=args.i_adapt_layer,
        adapt_layer_type=args.adapt_layer_type,
        adapt_enroll_dim=args.adapt_enroll_dim,
        n_blocks=args.n_blocks,
        n_repeats=args.n_repeats,
        bn_chan=args.bn_chan,
        hid_chan=args.hid_chan,
        skip_chan=args.skip_chan,
        conv_kernel_size=args.conv_kernel_size,
        norm_type=args.norm_type,
        mask_act=args.mask_act,
        fb_name=args.fb_name,
        kernel_size=args.kernel_size,
        n_filters=args.n_filters,
        stride=args.stride,
        sample_rate=args.sample_rate,
    )


def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    losses: list[float] = []
    scores: list[float] = []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        enrollment = batch["enrollment"].to(device)
        with torch.set_grad_enabled(train):
            est = model(mixture, enrollment)
            if est.dim() == 3:
                est = est.squeeze(1)
            loss = si_sdr_loss(est, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()) * mixture.shape[0])
        scores.append(float(si_sdr(est.detach(), target).mean().cpu()) * mixture.shape[0])
    n = max(len(loader.dataset), 1)
    return sum(losses) / n, sum(scores) / n


def evaluate(model, manifest: str, args: argparse.Namespace, device) -> dict[str, float]:
    ds = TSEDataset(manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=tse_collate)
    rows = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            mixture = batch["mixture"].to(device)
            target = batch["target"].to(device)
            enrollment = batch["enrollment"].to(device)
            est = model(mixture, enrollment)
            if est.dim() == 3:
                est = est.squeeze(1)
            row_src = ds.rows[i]
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

    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
        return sum(vals) / len(vals) if vals else float("nan")

    return {key: mean(key) for key in ["si_sdr", "si_sdri", "sdr", "sir", "sar", "stoi", "pesq"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--valid_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--n_filters", type=int, default=256)
    parser.add_argument("--kernel_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--fb_name", default="free")
    parser.add_argument("--n_blocks", type=int, default=6)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--bn_chan", type=int, default=64)
    parser.add_argument("--hid_chan", type=int, default=256)
    parser.add_argument("--skip_chan", type=int, default=64)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--norm_type", default="gLN")
    parser.add_argument("--mask_act", default="relu")
    parser.add_argument("--adapt_enroll_dim", type=int, default=64)
    parser.add_argument("--adapt_layer_type", default="mul")
    parser.add_argument("--i_adapt_layer", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_ds = TSEDataset(args.train_manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)
    valid_ds = TSEDataset(args.valid_manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=tse_collate,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tse_collate,
    )
    model = build_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    best_valid = float("inf")
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"resumed model={args.resume_checkpoint} start_epoch={start_epoch}", flush=True)

    history_path = exp_dir / "train_log.csv"
    mode = "a" if args.resume_checkpoint and history_path.exists() else "w"
    if mode == "a":
        with history_path.open("r", newline="", encoding="utf-8") as f:
            previous = list(csv.DictReader(f))
        if previous:
            best_valid = min(float(row["valid_loss"]) for row in previous)
    with history_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss", "valid_si_sdr"])
        if mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_sisdr = run_epoch(model, train_loader, optimizer, device, train=True)
            valid_loss, valid_sisdr = run_epoch(model, valid_loader, optimizer, device, train=False)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_si_sdr": valid_sisdr,
                }
            )
            f.flush()
            print(
                f"epoch={epoch} train={train_loss:.4f} train_si_sdr={train_sisdr:.3f} "
                f"valid={valid_loss:.4f} valid_si_sdr={valid_sisdr:.3f}",
                flush=True,
            )
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, exp_dir / "last.pt")
            if valid_loss < best_valid:
                best_valid = valid_loss
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, exp_dir / "best.pt")

    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    summary = evaluate(model, args.test_manifest, args, device)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
