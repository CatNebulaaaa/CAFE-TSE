from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from cafe_tse.utils.audio_io import fix_length, read_wav, write_wav
from cafe_tse.utils.manifest import read_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--speaker_pool_root", default=None)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--enrollment_seconds", type=float, default=3.0)
    parser.add_argument("--strategy", choices=["different_utterance", "source_crop"], default="source_crop")
    parser.add_argument("--out_dir", default="data/processed/enrollment")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_spk[str(row["speaker_id"])].append(i)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for i, row in enumerate(rows):
        sr = int(row["sample_rate"])
        length = int(sr * args.enrollment_seconds)
        enroll_source = row["target_path"]
        if args.strategy == "different_utterance":
            candidates = [j for j in by_spk[str(row["speaker_id"])] if j != i]
            if candidates:
                enroll_source = rows[candidates[0]]["target_path"]
        wav, _ = read_wav(enroll_source, target_sr=sr)
        wav = fix_length(wav, length)
        out_path = out_dir / f"{row['utt_id']}_enroll.wav"
        write_wav(out_path, wav, sr)
        new_paths.append(str(out_path))
    for row, path in zip(rows, new_paths):
        row["enrollment_path"] = path
        row["enrollment_length"] = args.enrollment_seconds
    write_manifest(rows, args.out_manifest)
    print(f"wrote {args.out_manifest}")


if __name__ == "__main__":
    main()
