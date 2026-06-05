from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from cafe_tse.utils.audio_io import fix_length, read_wav, write_wav
from cafe_tse.utils.manifest import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/metadata/minilibrimix_disjoint/test_manifest_final.csv")
    parser.add_argument("--out_dir", default="data/metadata/minilibrimix_disjoint/three_speaker_stress")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.manifest, newline="", encoding="utf-8")))
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    mix_dir = wav_dir / "mixture"
    int_dir = wav_dir / "interferer"
    mix_dir.mkdir(parents=True, exist_ok=True)
    int_dir.mkdir(parents=True, exist_ok=True)

    out_rows = []
    for i, row in enumerate(rows[: args.limit]):
        third = None
        for offset in range(1, len(rows)):
            candidate = rows[(i + offset) % len(rows)]
            if candidate["speaker_id"] != row["speaker_id"]:
                third = candidate
                break
        if third is None:
            continue

        target, _ = read_wav(row["target_path"], args.sample_rate)
        interferer1, _ = read_wav(row["interferer_path"], args.sample_rate)
        interferer2, _ = read_wav(third["interferer_path"], args.sample_rate)
        n = min(target.numel(), interferer1.numel(), interferer2.numel())
        n = min(n, int(args.sample_rate * 4.0))
        target = fix_length(target, n)
        interferer1 = fix_length(interferer1, n)
        interferer2 = fix_length(interferer2, n)
        interferer = interferer1 + interferer2
        mixture = target + interferer
        peak = torch.stack([mixture.abs().max(), target.abs().max(), interferer.abs().max()]).max().clamp_min(1.0)
        mixture = mixture / peak * 0.95
        interferer = interferer / peak * 0.95

        utt_id = f"three_{row['utt_id']}"
        mixture_path = mix_dir / f"{utt_id}.wav"
        interferer_path = int_dir / f"{utt_id}.wav"
        write_wav(mixture_path, mixture, args.sample_rate)
        write_wav(interferer_path, interferer, args.sample_rate)

        new_row = dict(row)
        new_row["utt_id"] = utt_id
        new_row["mixture_path"] = str(mixture_path)
        new_row["interferer_path"] = str(interferer_path)
        new_row["num_speakers"] = 3
        new_row["difficulty"] = "three_speaker"
        out_rows.append(new_row)

    out_manifest = out_dir / "test_manifest_final.csv"
    write_manifest(out_rows, out_manifest)
    print(f"wrote {out_manifest} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
