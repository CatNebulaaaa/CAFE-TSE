from __future__ import annotations

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
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def main() -> None:
    files = [
        ("Mixture", Path("results/example_mixture.wav")),
        ("Target", Path("results/example_target.wav")),
        ("Baseline", Path("results/example_baseline.wav")),
        ("5-block distill", Path("results/example_5block.wav")),
        ("5-block + EGSP", Path("results/example_egsp.wav")),
    ]
    fig, axes = plt.subplots(len(files), 1, figsize=(8.5, 9.0), constrained_layout=True)
    for ax, (title, path) in zip(axes, files):
        wav, sr = read_wav(path)
        ax.specgram(wav, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Hz")
        ax.set_ylim(0, sr / 2)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Spectrogram Case Study: Target Speaker Extraction", fontsize=13)
    out = Path("results/figures/spectrogram_case_study.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
