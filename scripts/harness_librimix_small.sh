#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
LIBRIMIX_ROOT="${LIBRIMIX_ROOT:-data/raw/LibriMix/Libri2Mix/wav16k/min}"

python -m cafe_tse.cli.prepare_librimix_manifest \
  --librimix_root "$LIBRIMIX_ROOT" \
  --out_dir data/metadata/librimix \
  --sample_rate 16000 \
  --num_speakers 2 \
  --mixture_type mix_clean \
  --max_train_samples 200 \
  --max_valid_samples 50 \
  --max_test_samples 50

python -m cafe_tse.cli.compute_complexity_manifest --manifest data/metadata/librimix/train_manifest.csv --out_manifest data/metadata/librimix/train_manifest_final.csv --sample_rate 16000
python -m cafe_tse.cli.compute_complexity_manifest --manifest data/metadata/librimix/valid_manifest.csv --out_manifest data/metadata/librimix/valid_manifest_final.csv --sample_rate 16000
python -m cafe_tse.cli.compute_complexity_manifest --manifest data/metadata/librimix/test_manifest.csv --out_manifest data/metadata/librimix/test_manifest_final.csv --sample_rate 16000

python -m cafe_tse.cli.train \
  --config configs/cafe_tse_dynamic.yaml \
  --train_manifest data/metadata/librimix/train_manifest_final.csv \
  --valid_manifest data/metadata/librimix/valid_manifest_final.csv \
  --exp_dir experiments/librimix_small \
  --override max_epochs=2

python -m cafe_tse.cli.evaluate \
  --config configs/cafe_tse_dynamic.yaml \
  --checkpoint experiments/librimix_small/checkpoints/best.pt \
  --test_manifest data/metadata/librimix/test_manifest_final.csv \
  --out_dir results/librimix_small \
  --save_audio 10

