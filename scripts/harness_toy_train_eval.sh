#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m cafe_tse.cli.prepare_toy_data --out_dir data/toy --num_samples 24 --sample_rate 8000 --duration 1.5
python -m cafe_tse.cli.compute_complexity_manifest --manifest data/toy/toy_manifest.csv --out_manifest data/toy/toy_manifest_complexity.csv --sample_rate 8000 --n_fft 128 --hop_length 32
python -m cafe_tse.cli.train --config configs/smoke_tiny.yaml --train_manifest data/toy/toy_manifest_complexity.csv --valid_manifest data/toy/toy_manifest_complexity.csv --exp_dir experiments/toy_train_eval --override max_epochs=3
python -m cafe_tse.cli.evaluate --config configs/smoke_tiny.yaml --checkpoint experiments/toy_train_eval/checkpoints/best.pt --test_manifest data/toy/toy_manifest_complexity.csv --out_dir results/toy_train_eval --save_audio 4 --device cpu

