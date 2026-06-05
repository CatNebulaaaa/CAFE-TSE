from __future__ import annotations

import csv
import json
from pathlib import Path

from cafe_tse.metrics.separation import compute_bss_metrics
from cafe_tse.utils.audio_io import fix_length, read_wav, rms_normalize


def main() -> None:
    manifest = Path("data/metadata/minilibrimix_disjoint/test_manifest_final.csv")
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    sr = 8000
    n = 32000
    acc: dict[str, float] = {}
    keys: list[str] | None = None
    same_path = 0
    for row in rows:
        if row["target_path"] == row["enrollment_path"]:
            same_path += 1
        mixture, _ = read_wav(row["mixture_path"], sr)
        target, _ = read_wav(row["target_path"], sr)
        enrollment, _ = read_wav(row["enrollment_path"], sr)
        interferer, _ = read_wav(row["interferer_path"], sr)
        mixture = rms_normalize(fix_length(mixture, n))
        target = rms_normalize(fix_length(target, n))
        enrollment = rms_normalize(fix_length(enrollment, n))
        interferer = rms_normalize(fix_length(interferer, n))
        metrics = compute_bss_metrics(enrollment, target, interferer, mixture, sr)
        keys = keys or list(metrics.keys())
        for key, value in metrics.items():
            acc[key] = acc.get(key, 0.0) + float(value)
    summary = {key: acc[key] / len(rows) for key in (keys or [])}
    summary["same_target_enrollment_paths"] = same_path
    summary["num_samples"] = len(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
