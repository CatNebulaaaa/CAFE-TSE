from __future__ import annotations

import csv

from pesq import pesq
from pystoi import stoi

from cafe_tse.utils.audio_io import fix_length, read_wav
from cafe_tse.metrics.separation import compute_bss_metrics, compute_basic_metrics


def main() -> None:
    sample_rate = 8000
    rows = list(csv.DictReader(open("data/metadata/minilibrimix_disjoint/test_manifest_final.csv", newline="", encoding="utf-8")))
    row = rows[0]
    target, _ = read_wav(row["target_path"], target_sr=sample_rate)
    mixture, _ = read_wav(row["mixture_path"], target_sr=sample_rate)
    ours, _ = read_wav(f"results/mini_exp18_egsp_spec_s005_selected/audio/{row['utt_id']}_estimated.wav", target_sr=sample_rate)
    length = 32000
    target_np = fix_length(target, length).numpy()
    for name, wav in [
        ("target", target),
        ("mixture", mixture),
        ("ours", ours),
    ]:
        wav_np = fix_length(wav, length).numpy()
        print(
            name,
            "stoi",
            stoi(target_np, wav_np, sample_rate, extended=False),
            "estoi",
            stoi(target_np, wav_np, sample_rate, extended=True),
            "pesq",
            pesq(sample_rate, target_np, wav_np, "nb"),
        )
    interferer, _ = read_wav(row["interferer_path"], target_sr=sample_rate)
    print("basic", compute_basic_metrics(ours, target, mixture, sample_rate))
    print("bss", compute_bss_metrics(ours, target, interferer, mixture, sample_rate))


if __name__ == "__main__":
    main()
