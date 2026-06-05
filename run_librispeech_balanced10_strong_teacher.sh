#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
PY=/root/miniconda3/envs/cafe-tse/bin/python
META=data/metadata/librispeech_tse_balanced10

"$PY" -m cafe_tse.cli.train \
  --config configs/base_usef_tfgridnet.yaml \
  --train_manifest "$META/train_manifest_final.csv" \
  --valid_manifest "$META/valid_manifest_final.csv" \
  --exp_dir experiments/mini_exp46_balanced10_strong_teacher \
  --override device=cuda \
  --override sample_rate=8000 \
  --override segment_seconds=4.0 \
  --override batch_size=4 \
  --override num_workers=0 \
  --override data.normalize_audio=true \
  --override data.curriculum=false \
  --override max_epochs=120 \
  --override early_stopping.patience=30 \
  --override early_stopping.min_delta=0.0002 \
  --override optim.lr=0.0005 \
  --override loss.si_sdr_weight=1.0 \
  --override loss.spectral_weight=0.05 \
  --override model.emb_dim=96 \
  --override model.hidden_dim=384 \
  --override model.n_blocks=8 \
  --override model.n_heads=8 \
  --override model.sparse_fusion_blocks='[0,1,2,3,4,5,6,7]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=8 \
  --override model.separator_output_mode=mag_mask

"$PY" -m cafe_tse.cli.evaluate \
  --config configs/base_usef_tfgridnet.yaml \
  --checkpoint experiments/mini_exp46_balanced10_strong_teacher/checkpoints/best.pt \
  --test_manifest "$META/test_manifest_final.csv" \
  --out_dir results/mini_exp46_balanced10_strong_teacher \
  --save_audio 12 \
  --device cuda \
  --override device=cuda \
  --override sample_rate=8000 \
  --override segment_seconds=4.0 \
  --override batch_size=4 \
  --override num_workers=0 \
  --override data.normalize_audio=true \
  --override model.emb_dim=96 \
  --override model.hidden_dim=384 \
  --override model.n_blocks=8 \
  --override model.n_heads=8 \
  --override model.sparse_fusion_blocks='[0,1,2,3,4,5,6,7]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=8 \
  --override model.separator_output_mode=mag_mask
