from __future__ import annotations

import argparse
import re
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    if sampwidth == 2:
        wav = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        wav = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sampwidth}: {path}")
    if channels > 1:
        wav = wav.reshape(-1, channels).mean(axis=1)
    return wav, sr


def discover_cases(audio_dir: Path) -> dict[str, dict[str, Path]]:
    cases: dict[str, dict[str, Path]] = {}
    pattern = re.compile(r"^(case\d+)_.+_(mixture|target|baseline|ours)\.wav$")
    for path in sorted(audio_dir.glob("case*.wav")):
        match = pattern.match(path.name)
        if not match:
            continue
        case_id, kind = match.groups()
        cases.setdefault(case_id, {})[kind] = path
    required = {"mixture", "target", "baseline", "ours"}
    return {case_id: files for case_id, files in cases.items() if required <= set(files)}


def draw_panel(
    rows: list[tuple[str, Path]],
    out: Path,
    title: str,
    figsize: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(len(rows), 1, figsize=figsize, constrained_layout=True)
    if len(rows) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, rows):
        wav, sr = read_wav(path)
        ax.specgram(wav, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
        ax.set_title(label, loc="left", fontsize=10)
        ax.set_ylabel("Hz")
        ax.set_ylim(0, min(8000, sr / 2))
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("demo_audio"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    cases = discover_cases(args.audio_dir)
    if not cases:
        raise SystemExit(f"No complete demo cases found in {args.audio_dir}")

    labels = [
        ("Mixture", "mixture"),
        ("Clean target", "target"),
        ("Baseline output", "baseline"),
        ("Final system output", "ours"),
    ]

    overview_rows: list[tuple[str, Path]] = []
    for case_id, files in sorted(cases.items()):
        rows = [(label, files[kind]) for label, kind in labels]
        draw_panel(
            rows,
            args.out_dir / f"spectrogram_{case_id}.png",
            f"Spectrogram Case Study: {case_id}",
            figsize=(8.5, 7.2),
        )
        overview_rows.extend((f"{case_id} - {label}", files[kind]) for label, kind in labels)

    draw_panel(
        overview_rows,
        args.out_dir / "spectrogram_demo_overview.png",
        "Spectrogram Overview Across Three Demo Cases",
        figsize=(8.5, 15.0),
    )
    print(f"wrote {len(cases) + 1} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
