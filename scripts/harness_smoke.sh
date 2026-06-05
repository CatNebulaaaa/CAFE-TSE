#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m cafe_tse.cli.prepare_toy_data \
  --out_dir data/toy \
  --num_samples 8 \
  --sample_rate 8000 \
  --duration 1.0 \
  --num_speakers 2

python -m cafe_tse.cli.compute_complexity_manifest \
  --manifest data/toy/toy_manifest.csv \
  --out_manifest data/toy/toy_manifest_complexity.csv \
  --sample_rate 8000 \
  --n_fft 128 \
  --hop_length 32

python -m cafe_tse.cli.train \
  --config configs/smoke_tiny.yaml \
  --train_manifest data/toy/toy_manifest_complexity.csv \
  --valid_manifest data/toy/toy_manifest_complexity.csv \
  --exp_dir experiments/smoke_tiny

python -m cafe_tse.cli.evaluate \
  --config configs/smoke_tiny.yaml \
  --checkpoint experiments/smoke_tiny/checkpoints/best.pt \
  --test_manifest data/toy/toy_manifest_complexity.csv \
  --out_dir results/smoke \
  --save_audio 2 \
  --device cpu

python -m cafe_tse.cli.infer \
  --config configs/smoke_tiny.yaml \
  --checkpoint experiments/smoke_tiny/checkpoints/best.pt \
  --mixture data/toy/mixtures/toy_000.wav \
  --enrollment data/toy/enrollments/toy_000.wav \
  --out_wav results/smoke/audio/estimated_0.wav \
  --device cpu

pytest tests/test_*.py -q

