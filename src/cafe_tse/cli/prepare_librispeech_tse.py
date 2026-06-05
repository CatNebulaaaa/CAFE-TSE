from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from cafe_tse.utils.audio_io import fix_length, read_wav, write_wav
from cafe_tse.utils.manifest import write_manifest


def _collect_audio(root: Path, min_files_per_speaker: int = 2) -> dict[str, list[Path]]:
    by_spk: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.flac")) + sorted(root.rglob("*.wav")):
        spk = path.name.split("-")[0]
        by_spk[spk].append(path)
    return {k: v for k, v in by_spk.items() if len(v) >= min_files_per_speaker}


def _scale_to_sir(target, interferer, sir_db: float):
    t_power = target.pow(2).mean().clamp_min(1e-8)
    i_power = interferer.pow(2).mean().clamp_min(1e-8)
    desired_i_power = t_power / (10 ** (sir_db / 10.0))
    return interferer * (desired_i_power / i_power).sqrt()


def _difficulty(i: int) -> tuple[str, float, float]:
    stage = i % 3
    if stage == 0:
        return "easy", 5.0, 30.0
    if stage == 1:
        return "medium", 0.0, 20.0
    return "hard", -5.0, 10.0


def _make_rows(
    roots: list[Path],
    out_dir: Path,
    split: str,
    num_samples: int,
    sample_rate: int,
    duration: float,
    seed: int,
    min_files_per_speaker: int = 2,
    max_speakers: int | None = None,
    shared_speaker_splits: bool = False,
    disable_noise: bool = False,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_spk: dict[str, list[Path]] = {}
    for root in roots:
        by_spk.update(_collect_audio(root, min_files_per_speaker))
    speakers = sorted(by_spk)
    if len(speakers) < 2:
        raise RuntimeError(f"Need at least two speakers under {roots}, found {len(speakers)}")
    if max_speakers:
        speakers = speakers[:max_speakers]
        by_spk = {spk: by_spk[spk] for spk in speakers}

    n = int(sample_rate * duration)
    mix_dir = out_dir / split / "mixture"
    tgt_dir = out_dir / split / "target"
    enr_dir = out_dir / split / "enrollment"
    int_dir = out_dir / split / "interferer"
    for d in [mix_dir, tgt_dir, enr_dir, int_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(num_samples):
        spk_a = speakers[i % len(speakers)] if shared_speaker_splits else rng.choice(speakers)
        candidates = [spk for spk in speakers if spk != spk_a]
        spk_b = rng.choice(candidates)
        target_pool = _split_pool(by_spk[spk_a], split) if shared_speaker_splits else by_spk[spk_a]
        interferer_pool = _split_pool(by_spk[spk_b], split) if shared_speaker_splits else by_spk[spk_b]
        if not target_pool:
            target_pool = by_spk[spk_a]
        if not interferer_pool:
            interferer_pool = by_spk[spk_b]
        target_path = rng.choice(target_pool)
        enroll_candidates = [p for p in target_pool if p != target_path]
        if not enroll_candidates:
            enroll_candidates = [p for p in by_spk[spk_a] if p != target_path]
        enroll_path = rng.choice(enroll_candidates)
        interferer_path = rng.choice(interferer_pool)
        target, _ = read_wav(target_path, target_sr=sample_rate)
        enrollment, _ = read_wav(enroll_path, target_sr=sample_rate)
        interferer, _ = read_wav(interferer_path, target_sr=sample_rate)
        target = _crop_or_pad(target, n, rng)
        enrollment = _crop_or_pad(enrollment, n, rng)
        interferer = _crop_or_pad(interferer, n, rng)
        difficulty, sir, snr = _difficulty(i)
        interferer = _scale_to_sir(target, interferer, sir)
        mixture = target + interferer
        if disable_noise:
            snr = float("inf")
        if math.isfinite(snr):
            sig_power = mixture.pow(2).mean().clamp_min(1e-8)
            noise_power = sig_power / (10 ** (snr / 10.0))
            noise = torch_randn_like(mixture, rng) * noise_power.sqrt()
            mixture = mixture + noise

        peak = max(float(mixture.abs().max()), float(target.abs().max()), 1.0)
        mixture = mixture / peak * 0.95
        target = target / peak * 0.95
        interferer = interferer / peak * 0.95
        enrollment = enrollment / max(float(enrollment.abs().max()), 1.0) * 0.95

        utt = f"{split}_{i:05d}"
        mix_path = mix_dir / f"{utt}_mixture.wav"
        tgt_path = tgt_dir / f"{utt}_target.wav"
        enr_path = enr_dir / f"{utt}_enrollment.wav"
        int_path = int_dir / f"{utt}_interferer.wav"
        write_wav(mix_path, mixture, sample_rate)
        write_wav(tgt_path, target, sample_rate)
        write_wav(enr_path, enrollment, sample_rate)
        write_wav(int_path, interferer, sample_rate)
        rows.append(
            {
                "utt_id": utt,
                "mixture_path": str(mix_path),
                "target_path": str(tgt_path),
                "enrollment_path": str(enr_path),
                "interferer_path": str(int_path),
                "target_source_path": str(target_path),
                "enrollment_source_path": str(enroll_path),
                "interferer_source_path": str(interferer_path),
                "speaker_id": spk_a,
                "split": split,
                "sample_rate": sample_rate,
                "duration": duration,
                "num_speakers": 2,
                "sir": sir,
                "snr": snr,
                "overlap_ratio": 1.0,
                "gender_condition": "unknown",
                "enrollment_length": duration,
                "enrollment_noise": "clean",
                "difficulty": difficulty,
                "complexity_score": 0.5,
            }
        )
    return rows


def _split_pool(paths: list[Path], split: str) -> list[Path]:
    n = len(paths)
    train_end = max(2, int(n * 0.70))
    valid_end = max(train_end + 1, int(n * 0.85))
    if split == "train":
        return paths[:train_end]
    if split == "valid":
        return paths[train_end:valid_end]
    if split == "test":
        return paths[valid_end:]
    return paths


def _crop_or_pad(wav, length: int, rng: np.random.Generator):
    if wav.numel() > length:
        start = int(rng.integers(0, wav.numel() - length + 1))
        return wav[start : start + length]
    return fix_length(wav, length)


def torch_randn_like(ref, rng: np.random.Generator):
    import torch

    arr = rng.standard_normal(ref.numel()).astype(np.float32)
    return torch.from_numpy(arr).reshape_as(ref).to(dtype=ref.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_roots", nargs="+", required=True)
    parser.add_argument("--valid_roots", nargs="+", default=None)
    parser.add_argument("--test_roots", nargs="+", required=True)
    parser.add_argument("--out_dir", default="data/librispeech_tse")
    parser.add_argument("--metadata_dir", default="data/metadata/librispeech_tse")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--num_train", type=int, default=600)
    parser.add_argument("--num_valid", type=int, default=120)
    parser.add_argument("--num_test", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_files_per_speaker", type=int, default=2)
    parser.add_argument("--max_speakers", type=int, default=None)
    parser.add_argument("--shared_speaker_splits", action="store_true")
    parser.add_argument("--disable_noise", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    metadata = Path(args.metadata_dir)
    metadata.mkdir(parents=True, exist_ok=True)
    train_roots = [Path(p) for p in args.train_roots]
    valid_roots_arg = [Path(p) for p in args.valid_roots] if args.valid_roots else None
    test_roots = [Path(p) for p in args.test_roots]

    common = {
        "min_files_per_speaker": args.min_files_per_speaker,
        "max_speakers": args.max_speakers,
        "shared_speaker_splits": args.shared_speaker_splits,
        "disable_noise": args.disable_noise,
    }
    train_rows = _make_rows(train_roots, out_dir, "train", args.num_train, args.sample_rate, args.duration, args.seed, **common)
    valid_roots = train_roots if args.shared_speaker_splits else (valid_roots_arg or train_roots)
    test_roots = train_roots if args.shared_speaker_splits else test_roots
    valid_rows = _make_rows(valid_roots, out_dir, "valid", args.num_valid, args.sample_rate, args.duration, args.seed + 1, **common)
    test_rows = _make_rows(test_roots, out_dir, "test", args.num_test, args.sample_rate, args.duration, args.seed + 2, **common)

    write_manifest(train_rows, metadata / "train_manifest.csv")
    write_manifest(valid_rows, metadata / "valid_manifest.csv")
    write_manifest(test_rows, metadata / "test_manifest.csv")
    print(f"wrote {metadata}")


if __name__ == "__main__":
    main()
