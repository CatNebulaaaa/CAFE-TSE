from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cafe_tse.datasets.collate import tse_collate
from cafe_tse.datasets.curriculum_sampler import CurriculumSchedule
from cafe_tse.datasets.tse_dataset import TSEDataset
from cafe_tse.engine.checkpoint import load_checkpoint, save_checkpoint
from cafe_tse.losses.sisdr import si_sdr, si_sdr_loss
from cafe_tse.losses.spectral import multi_resolution_spectral_l1_loss, spectral_l1_loss
from cafe_tse.models.cafe_tse import build_model
from cafe_tse.utils.config import apply_overrides, load_config, select_device
from cafe_tse.utils.logger import get_logger
from cafe_tse.utils.seed import seed_everything


class Trainer:
    def __init__(self, cfg: dict, train_manifest: str, valid_manifest: str, exp_dir: str):
        self.cfg = cfg
        seed_everything(int(cfg.get("seed", 42)))
        self.log = get_logger("cafe_tse.train")
        self.device = torch.device(select_device(cfg.get("device", "cuda")))
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        sr = int(cfg["sample_rate"])
        segment = float(cfg.get("segment_seconds", 4.0))
        normalize = bool(cfg.get("data", {}).get("normalize_audio", True))
        self.train_ds = TSEDataset(train_manifest, sr, segment, normalize)
        self.valid_ds = TSEDataset(valid_manifest, sr, segment, normalize)
        self.model = build_model(cfg).to(self.device)
        self.teacher = None
        self.distill_cfg = cfg.get("distill", {})
        if self.distill_cfg.get("enabled"):
            teacher_config = self.distill_cfg.get("teacher_config")
            teacher_checkpoint = self.distill_cfg.get("teacher_checkpoint")
            if not teacher_config or not teacher_checkpoint:
                raise ValueError("distill.enabled requires distill.teacher_config and distill.teacher_checkpoint")
            teacher_overrides = list(self.distill_cfg.get("teacher_overrides", []) or [])
            teacher_overrides.extend(
                [
                    f"device={cfg.get('device', 'cuda')}",
                    f"sample_rate={cfg['sample_rate']}",
                    f"segment_seconds={cfg.get('segment_seconds', 4.0)}",
                    f"model.n_fft={cfg['model'].get('n_fft', 512)}",
                    f"model.hop_length={cfg['model'].get('hop_length', 128)}",
                    f"model.win_length={cfg['model'].get('win_length', cfg['model'].get('n_fft', 512))}",
                ]
            )
            teacher_cfg = apply_overrides(load_config(str(teacher_config)), teacher_overrides)
            self.teacher = build_model(teacher_cfg).to(self.device)
            load_checkpoint(str(teacher_checkpoint), self.teacher, map_location=str(self.device))
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.get("optim", {}).get("lr", 3e-4)),
            weight_decay=float(cfg.get("optim", {}).get("weight_decay", 1e-5)),
        )
        self.start_epoch = 1
        init_checkpoint = cfg.get("init_checkpoint")
        if init_checkpoint:
            strict = bool(cfg.get("init_strict", True))
            load_checkpoint(init_checkpoint, self.model, optimizer=None, map_location=str(self.device), strict=strict)
            self.log.info("initialized checkpoint=%s strict=%s", init_checkpoint, strict)
        resume_checkpoint = cfg.get("resume_checkpoint")
        if resume_checkpoint:
            ckpt = load_checkpoint(resume_checkpoint, self.model, self.optimizer, map_location=str(self.device))
            self.start_epoch = int(ckpt.get("epoch", 0)) + 1
            self.log.info("resumed checkpoint=%s start_epoch=%s", resume_checkpoint, self.start_epoch)
        self.schedule = None
        if cfg.get("data", {}).get("curriculum"):
            self.schedule = CurriculumSchedule(cfg["data"].get("curriculum_schedule", []))

    def _loader(self, ds: TSEDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=int(self.cfg.get("batch_size", 4)),
            shuffle=shuffle,
            num_workers=int(self.cfg.get("num_workers", 0)),
            collate_fn=tse_collate,
        )

    def _loss(self, est: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_cfg = self.cfg.get("loss", {})
        loss = float(loss_cfg.get("si_sdr_weight", 1.0)) * si_sdr_loss(est, target)
        spectral_weight = float(loss_cfg.get("spectral_weight", 0.0))
        if spectral_weight:
            loss = loss + spectral_weight * spectral_l1_loss(
                est,
                target,
                n_fft=int(self.cfg["model"].get("n_fft", 512)),
                hop_length=int(self.cfg["model"].get("hop_length", 128)),
            )
        mr_weight = float(loss_cfg.get("multi_resolution_spectral_weight", 0.0))
        if mr_weight:
            fft_sizes = loss_cfg.get("multi_resolution_fft_sizes", [256, 512, 1024])
            loss = loss + mr_weight * multi_resolution_spectral_l1_loss(est, target, fft_sizes=fft_sizes)
        return loss

    def _distill_loss(self, est: torch.Tensor, teacher_wav: torch.Tensor) -> torch.Tensor:
        teacher_weight = float(self.distill_cfg.get("teacher_weight", 0.0))
        if not teacher_weight:
            return est.new_tensor(0.0)
        loss = teacher_weight * si_sdr_loss(est, teacher_wav)
        l1_weight = float(self.distill_cfg.get("teacher_l1_weight", 0.0))
        if l1_weight:
            loss = loss + l1_weight * F.l1_loss(est, teacher_wav)
        spectral_weight = float(self.distill_cfg.get("teacher_spectral_weight", 0.0))
        if spectral_weight:
            loss = loss + spectral_weight * spectral_l1_loss(
                est,
                teacher_wav,
                n_fft=int(self.cfg["model"].get("n_fft", 512)),
                hop_length=int(self.cfg["model"].get("hop_length", 128)),
            )
        return loss

    def fit(self) -> dict:
        history_path = self.exp_dir / "train_log.csv"
        fieldnames = ["epoch", "train_loss", "valid_loss", "valid_si_sdr", "difficulties"]
        best = float("inf")
        stale_epochs = 0
        previous_rows = []
        if self.start_epoch > 1 and history_path.exists():
            with history_path.open("r", newline="", encoding="utf-8") as f:
                previous_rows = list(csv.DictReader(f))
            if previous_rows:
                valid_losses = [float(row["valid_loss"]) for row in previous_rows]
                best = min(valid_losses)
                best_idx = valid_losses.index(best)
                stale_epochs = len(valid_losses) - best_idx - 1
        mode = "a" if self.start_epoch > 1 and history_path.exists() else "w"
        with history_path.open(mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            elif previous_rows:
                self.log.info(
                    "resuming history=%s previous_epochs=%s best_valid=%.4f stale_epochs=%s",
                    history_path,
                    len(previous_rows),
                    best,
                    stale_epochs,
                )
            stop_cfg = self.cfg.get("early_stopping", {})
            patience = int(stop_cfg.get("patience", 0) or 0)
            min_delta = float(stop_cfg.get("min_delta", 0.0) or 0.0)
            for epoch in range(self.start_epoch, int(self.cfg.get("max_epochs", 1)) + 1):
                difficulties = ["easy", "medium", "hard"]
                if self.schedule:
                    difficulties = self.schedule.allowed_difficulties(epoch)
                    self.train_ds.set_allowed_difficulties(difficulties)
                train_loss = self._run_epoch(self._loader(self.train_ds, True), train=True)
                valid_loss, valid_si_sdr = self._validate()
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "valid_si_sdr": valid_si_sdr,
                    "difficulties": ",".join(difficulties),
                }
                writer.writerow(row)
                f.flush()
                self.log.info("epoch=%s train=%.4f valid=%.4f si_sdr=%.3f difficulties=%s", epoch, train_loss, valid_loss, valid_si_sdr, difficulties)
                save_checkpoint(self.ckpt_dir / "last.pt", self.model, self.optimizer, epoch, row)
                if valid_loss < best - min_delta:
                    best = valid_loss
                    stale_epochs = 0
                    save_checkpoint(self.ckpt_dir / "best.pt", self.model, self.optimizer, epoch, row)
                else:
                    stale_epochs += 1
                    if patience and stale_epochs >= patience:
                        self.log.info("early stopping at epoch=%s best_valid=%.4f patience=%s", epoch, best, patience)
                        break
        return {"best_valid_loss": best}

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total = 0.0
        count = 0
        for batch in loader:
            mixture = batch["mixture"].to(self.device)
            target = batch["target"].to(self.device)
            enrollment = batch["enrollment"].to(self.device)
            with torch.set_grad_enabled(train):
                teacher_wav = None
                if train and self.teacher is not None:
                    with torch.no_grad():
                        teacher_wav = self.teacher(mixture, enrollment).wav.detach()
                active_blocks_override = None
                depth_cfg = self.cfg.get("depth_aware", {})
                if train and depth_cfg.get("enabled"):
                    depths = list(depth_cfg.get("active_depths", []) or [])
                    if depths:
                        probs = depth_cfg.get("probabilities")
                        sampled = torch.multinomial(
                            torch.tensor(probs or [1.0] * len(depths), dtype=torch.float),
                            num_samples=mixture.shape[0],
                            replacement=True,
                        )
                        active_blocks_override = [int(depths[int(idx)]) for idx in sampled.tolist()]
                out = self.model(mixture, enrollment, active_blocks_override=active_blocks_override)
                loss = self._loss(out.wav, target)
                if teacher_wav is not None:
                    loss = loss + self._distill_loss(out.wav, teacher_wav)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.get("optim", {}).get("grad_clip", 5.0)))
                    self.optimizer.step()
            total += float(loss.detach().cpu()) * mixture.shape[0]
            count += mixture.shape[0]
        return total / max(count, 1)

    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        losses = []
        scores = []
        with torch.no_grad():
            for batch in self._loader(self.valid_ds, False):
                mixture = batch["mixture"].to(self.device)
                target = batch["target"].to(self.device)
                enrollment = batch["enrollment"].to(self.device)
                out = self.model(mixture, enrollment)
                losses.append(float(self._loss(out.wav, target).cpu()) * mixture.shape[0])
                scores.append(float(si_sdr(out.wav, target).mean().cpu()) * mixture.shape[0])
        n = max(len(self.valid_ds), 1)
        return sum(losses) / n, sum(scores) / n
