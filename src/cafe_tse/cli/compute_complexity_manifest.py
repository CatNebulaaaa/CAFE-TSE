from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from cafe_tse.features.complexity import compute_complexity_score
from cafe_tse.utils.audio_io import read_wav
from cafe_tse.utils.manifest import read_manifest, write_manifest


def difficulty_from_score(score: float) -> str:
    if score < 0.4:
        return "easy"
    if score < 0.65:
        return "medium"
    return "hard"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--difficulty_rule", default="curriculum_v1")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    scores: list[float] = []
    difficulties: list[str] = []
    for row in tqdm(rows, total=len(rows), desc="complexity"):
        wav, _ = read_wav(row["mixture_path"], target_sr=args.sample_rate)
        with torch.no_grad():
            score = float(compute_complexity_score(wav.unsqueeze(0), args.sample_rate, args.n_fft, args.hop_length).item())
        scores.append(score)
        if args.difficulty_rule == "keep" and isinstance(row.get("difficulty"), str):
            difficulties.append(row["difficulty"])
        else:
            difficulties.append(difficulty_from_score(score))
    for row, score, difficulty in zip(rows, scores, difficulties):
        row["complexity_score"] = score
        row["difficulty"] = difficulty
    write_manifest(rows, args.out_manifest)
    print(f"wrote {args.out_manifest}")


if __name__ == "__main__":
    main()
