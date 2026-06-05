from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "asteroid_site"))
sys.path.insert(0, str(ROOT / "third_party" / "speakerbeam" / "src"))

from train_open_speakerbeam import build_model, evaluate  # noqa: E402

from cafe_tse.datasets.collate import tse_collate  # noqa: E402
from cafe_tse.datasets.tse_dataset import TSEDataset  # noqa: E402
from cafe_tse.losses.sisdr import si_sdr, si_sdr_loss  # noqa: E402


def run_epoch(student, teacher, loader, optimizer, device, args, train: bool) -> tuple[float, float]:
    student.train(train)
    teacher.eval()
    losses: list[float] = []
    scores: list[float] = []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        enrollment = batch["enrollment"].to(device)
        with torch.no_grad():
            teacher_est = teacher(mixture, enrollment)
            if teacher_est.dim() == 3:
                teacher_est = teacher_est.squeeze(1)
        with torch.set_grad_enabled(train):
            est = student(mixture, enrollment)
            if est.dim() == 3:
                est = est.squeeze(1)
            loss = args.target_weight * si_sdr_loss(est, target)
            loss = loss + args.teacher_weight * si_sdr_loss(est, teacher_est)
            if args.teacher_l1_weight:
                loss = loss + args.teacher_l1_weight * F.l1_loss(est, teacher_est)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()) * mixture.shape[0])
        scores.append(float(si_sdr(est.detach(), target).mean().cpu()) * mixture.shape[0])
    n = max(len(loader.dataset), 1)
    return sum(losses) / n, sum(scores) / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--valid_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_weight", type=float, default=1.0)
    parser.add_argument("--teacher_weight", type=float, default=0.5)
    parser.add_argument("--teacher_l1_weight", type=float, default=0.0)
    parser.add_argument("--n_filters", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--fb_name", default="free")
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--bn_chan", type=int, default=32)
    parser.add_argument("--hid_chan", type=int, default=128)
    parser.add_argument("--skip_chan", type=int, default=32)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--norm_type", default="gLN")
    parser.add_argument("--mask_act", default="relu")
    parser.add_argument("--adapt_enroll_dim", type=int, default=32)
    parser.add_argument("--adapt_layer_type", default="mul")
    parser.add_argument("--i_adapt_layer", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_args = argparse.Namespace(**teacher_ckpt["args"])
    teacher = build_model(teacher_args).to(device)
    teacher.load_state_dict(teacher_ckpt["model"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = build_model(args).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    train_ds = TSEDataset(args.train_manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)
    valid_ds = TSEDataset(args.valid_manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=tse_collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=tse_collate)

    best_valid = float("inf")
    with (exp_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss", "valid_si_sdr"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_sisdr = run_epoch(student, teacher, train_loader, optimizer, device, args, train=True)
            valid_loss, valid_sisdr = run_epoch(student, teacher, valid_loader, optimizer, device, args, train=False)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_si_sdr": valid_sisdr})
            f.flush()
            print(f"epoch={epoch} train={train_loss:.4f} train_si_sdr={train_sisdr:.3f} valid={valid_loss:.4f} valid_si_sdr={valid_sisdr:.3f}", flush=True)
            torch.save({"model": student.state_dict(), "args": vars(args), "epoch": epoch}, exp_dir / "last.pt")
            if valid_loss < best_valid:
                best_valid = valid_loss
                torch.save({"model": student.state_dict(), "args": vars(args), "epoch": epoch}, exp_dir / "best.pt")

    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    student.load_state_dict(ckpt["model"])
    summary = evaluate(student, args.test_manifest, args, device)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
