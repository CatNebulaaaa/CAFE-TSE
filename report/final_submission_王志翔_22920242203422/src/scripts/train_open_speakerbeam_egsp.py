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
sys.path.insert(0, str(ROOT / "scripts"))

from train_open_speakerbeam import build_model, evaluate  # noqa: E402

from cafe_tse.datasets.collate import tse_collate  # noqa: E402
from cafe_tse.datasets.tse_dataset import TSEDataset  # noqa: E402
from cafe_tse.losses.sisdr import si_sdr, si_sdr_loss  # noqa: E402


def egsp_filter(mixture: torch.Tensor, enrollment: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.egsp_strength <= 0:
        return mixture
    length = mixture.shape[-1]
    window = torch.hann_window(args.egsp_n_fft, device=mixture.device)
    mix_spec = torch.stft(mixture, n_fft=args.egsp_n_fft, hop_length=args.egsp_hop, window=window, return_complex=True)
    enroll_spec = torch.stft(enrollment, n_fft=args.egsp_n_fft, hop_length=args.egsp_hop, window=window, return_complex=True)
    profile = enroll_spec.abs().mean(dim=-1, keepdim=True)
    profile = profile / profile.mean(dim=1, keepdim=True).clamp_min(1e-6)
    weight = 1.0 + args.egsp_strength * (profile - 1.0)
    weight = weight.clamp(args.egsp_min_weight, args.egsp_max_weight)
    enhanced = torch.istft(mix_spec * weight, n_fft=args.egsp_n_fft, hop_length=args.egsp_hop, window=window, length=length)
    return enhanced


def run_epoch(model, loader, optimizer, device, args, train: bool) -> tuple[float, float]:
    model.train(train)
    losses: list[float] = []
    scores: list[float] = []
    for batch in loader:
        mixture = batch["mixture"].to(device)
        target = batch["target"].to(device)
        enrollment = batch["enrollment"].to(device)
        mixture_in = egsp_filter(mixture, enrollment, args)
        with torch.set_grad_enabled(train):
            est = model(mixture_in, enrollment)
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


def evaluate_egsp(model, manifest: str, args: argparse.Namespace, device) -> dict[str, float]:
    class EgspWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, mixture, enrollment):
            return self.base(egsp_filter(mixture, enrollment, args), enrollment)

    return evaluate(EgspWrapper(model).to(device), manifest, args, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--valid_manifest", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--segment_seconds", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--egsp_strength", type=float, default=0.05)
    parser.add_argument("--egsp_n_fft", type=int, default=256)
    parser.add_argument("--egsp_hop", type=int, default=64)
    parser.add_argument("--egsp_min_weight", type=float, default=0.85)
    parser.add_argument("--egsp_max_weight", type=float, default=1.15)
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
    model = build_model(args).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"initialized checkpoint={args.init_checkpoint}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_ds = TSEDataset(args.train_manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)
    valid_ds = TSEDataset(args.valid_manifest, args.sample_rate, args.segment_seconds, normalize_audio=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=tse_collate)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=tse_collate)

    best_valid_sisdr = float("-inf")
    with (exp_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss", "valid_si_sdr"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_sisdr = run_epoch(model, train_loader, optimizer, device, args, train=True)
            valid_loss, valid_sisdr = run_epoch(model, valid_loader, optimizer, device, args, train=False)
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss, "valid_si_sdr": valid_sisdr})
            f.flush()
            print(f"epoch={epoch} train={train_loss:.4f} train_si_sdr={train_sisdr:.3f} valid={valid_loss:.4f} valid_si_sdr={valid_sisdr:.3f}", flush=True)
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, exp_dir / "last.pt")
            if valid_sisdr > best_valid_sisdr:
                best_valid_sisdr = valid_sisdr
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, exp_dir / "best.pt")

    ckpt = torch.load(exp_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    summary = evaluate_egsp(model, args.test_manifest, args, device)
    (exp_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
