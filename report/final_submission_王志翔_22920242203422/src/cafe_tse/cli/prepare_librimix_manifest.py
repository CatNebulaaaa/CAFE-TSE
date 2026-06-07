from __future__ import annotations

import argparse
from pathlib import Path

from cafe_tse.utils.audio_io import read_wav
from cafe_tse.utils.manifest import write_manifest


SPLITS = {"train": "train-100", "valid": "dev", "test": "test"}
MINI_SPLITS = {"train": "train", "valid": "val", "test": "val"}


def _slice(rows: list[dict], limit: int | None, offset: int = 0) -> list[dict]:
    rows = rows[offset:]
    return rows if not limit else rows[:limit]


def _speaker_id_from_source(stem: str, source_index: int = 0) -> str:
    parts = stem.split("_")
    source_stem = parts[min(source_index, len(parts) - 1)]
    return source_stem.split("-")[0]


def _build_enrollment_pool(source_dirs: list[Path]) -> dict[str, list[Path]]:
    by_speaker: dict[str, list[Path]] = {}
    for source_index, source_dir in enumerate(source_dirs):
        for path in sorted(source_dir.glob("*.wav")):
            by_speaker.setdefault(_speaker_id_from_source(path.stem, source_index), []).append(path)
    return by_speaker


def _assign_disjoint_enrollments(
    rows: list[dict],
    enrollment_pool: dict[str, list[Path]],
    drop_singles: bool = False,
) -> list[dict]:
    used_by_speaker: dict[str, int] = {}

    assigned = []
    for row in rows:
        alternatives = [p for p in enrollment_pool.get(row["speaker_id"], []) if str(p) != row["target_path"]]
        if alternatives:
            row = dict(row)
            start = used_by_speaker.get(row["speaker_id"], 0)
            row["enrollment_path"] = str(alternatives[start % len(alternatives)])
            used_by_speaker[row["speaker_id"]] = start + 1
            row["enrollment_noise"] = "same_speaker_disjoint"
            assigned.append(row)
        elif not drop_singles:
            assigned.append(row)
    return assigned


def build_split(
    root: Path,
    split: str,
    mixture_type: str,
    sample_rate: int,
    num_speakers: int,
    limit: int | None,
    offset: int = 0,
    disjoint_enrollment: bool = False,
    drop_single_enrollment_speakers: bool = False,
) -> list[dict]:
    split_name = SPLITS[split]
    split_dir = root / split_name
    if not split_dir.exists():
        split_dir = root / MINI_SPLITS[split]
    mix_dir = split_dir / mixture_type
    source_dirs = [split_dir / f"s{i + 1}" for i in range(num_speakers)]
    enrollment_pool = _build_enrollment_pool(source_dirs)
    rows = []
    for mix_path in sorted(mix_dir.glob("*.wav")):
        stem = mix_path.name
        target_path = source_dirs[0] / stem
        interferer_path = source_dirs[1] / stem if len(source_dirs) > 1 else ""
        if not target_path.exists():
            continue
        speaker_id = _speaker_id_from_source(target_path.stem)
        enrollment_path = target_path
        try:
            wav, sr = read_wav(mix_path, target_sr=None)
            duration = wav.numel() / float(sr)
        except Exception:
            duration = 0.0
        rows.append(
            {
                "utt_id": f"{split}_{mix_path.stem}",
                "mixture_path": str(mix_path),
                "target_path": str(target_path),
                "enrollment_path": str(enrollment_path),
                "interferer_path": str(interferer_path) if interferer_path and interferer_path.exists() else "",
                "speaker_id": speaker_id,
                "split": split,
                "sample_rate": sample_rate,
                "duration": duration,
                "num_speakers": num_speakers,
                "sir": 0.0,
                "snr": 0.0,
                "overlap_ratio": 1.0,
                "gender_condition": "unknown",
                "enrollment_length": duration,
                "enrollment_noise": "source_crop",
                "difficulty": ["easy", "medium", "hard"][len(rows) % 3],
                "complexity_score": 0.5,
            }
        )
    if disjoint_enrollment:
        rows = _assign_disjoint_enrollments(rows, enrollment_pool, drop_single_enrollment_speakers)
    return _slice(rows, limit, offset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librimix_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--mixture_type", default="mix_clean")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--valid_offset", type=int, default=0)
    parser.add_argument("--test_offset", type=int, default=0)
    parser.add_argument("--disjoint_enrollment", action="store_true")
    parser.add_argument("--drop_single_enrollment_speakers", action="store_true")
    args = parser.parse_args()

    root = Path(args.librimix_root)
    out = Path(args.out_dir)
    limits = {"train": args.max_train_samples, "valid": args.max_valid_samples, "test": args.max_test_samples}
    offsets = {"train": 0, "valid": args.valid_offset, "test": args.test_offset}
    for split in ["train", "valid", "test"]:
        rows = build_split(
            root,
            split,
            args.mixture_type,
            args.sample_rate,
            args.num_speakers,
            limits[split],
            offsets[split],
            args.disjoint_enrollment,
            args.drop_single_enrollment_speakers,
        )
        write_manifest(rows, out / f"{split}_manifest.csv")
        print(f"wrote {out / f'{split}_manifest.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
