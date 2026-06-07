from __future__ import annotations

import torch


def tse_collate(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "utt_id": [str(x["utt_id"]) for x in batch],
        "mixture": torch.stack([x["mixture"] for x in batch]),  # type: ignore[list-item]
        "target": torch.stack([x["target"] for x in batch]),  # type: ignore[list-item]
        "enrollment": torch.stack([x["enrollment"] for x in batch]),  # type: ignore[list-item]
        "complexity_score": torch.stack([x["complexity_score"] for x in batch]),  # type: ignore[list-item]
        "difficulty": [str(x["difficulty"]) for x in batch],
        "sample_rate": [int(x["sample_rate"]) for x in batch],
        "norm_gain": torch.stack([x["norm_gain"] for x in batch]),  # type: ignore[list-item]
    }
