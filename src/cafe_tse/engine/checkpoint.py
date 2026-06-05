from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, model, optimizer=None, map_location: str = "cpu", strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
