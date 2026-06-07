from __future__ import annotations

from pathlib import Path
import csv

MANIFEST_COLUMNS = [
    "utt_id",
    "mixture_path",
    "target_path",
    "enrollment_path",
    "speaker_id",
    "split",
    "sample_rate",
    "duration",
    "num_speakers",
    "sir",
    "snr",
    "overlap_ratio",
    "gender_condition",
    "enrollment_length",
    "enrollment_noise",
    "difficulty",
    "complexity_score",
]


def _default(col: str):
    if col == "complexity_score":
        return 0.5
    if col == "difficulty":
        return "medium"
    if col in {"sir", "snr", "overlap_ratio", "enrollment_length", "duration"}:
        return 0.0
    if col == "sample_rate":
        return 16000
    if col == "num_speakers":
        return 2
    return ""


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for col in MANIFEST_COLUMNS:
            row.setdefault(col, _default(col))
    return rows


def write_manifest(rows, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    rows = list(rows)
    extra = []
    for row in rows:
        for key in row:
            if key not in MANIFEST_COLUMNS and key not in extra:
                extra.append(key)
    cols = MANIFEST_COLUMNS + extra
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
