from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent_name = cfg.pop("extends", None)
    if parent_name:
        parent_path = path.parent / parent_name
        parent = load_config(parent_path)
        cfg = _deep_merge(parent, cfg)
    return cfg


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    out = deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = out
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def select_device(requested: str | None) -> str:
    if requested is None:
        return "cuda"
    if requested == "cuda":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return requested

