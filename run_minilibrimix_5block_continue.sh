#!/usr/bin/env bash
set -euo pipefail

MAX_EPOCHS=${1:-160}
PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

"$PY" -m cafe_tse.cli.train \
  --config configs/cafe_tse_dynamic.yaml \
  --train_manifest data/metadata/minilibrimix_disjoint/train_manifest_final.csv \
  --valid_manifest data/metadata/minilibrimix_disjoint/valid_manifest_final.csv \
  --exp_dir experiments/mini_exp10_distill_5block_mid \
  --override device=cuda \
  --override sample_rate=8000 \
  --override segment_seconds=4.0 \
  --override batch_size=8 \
  --override num_workers=0 \
  --override max_epochs="$MAX_EPOCHS" \
  --override early_stopping.patience=12 \
  --override early_stopping.min_delta=0.0005 \
  --override model.n_fft=512 \
  --override model.hop_length=128 \
  --override model.win_length=512 \
  --override loss.spectral_weight=0.1 \
  --override distill.enabled=true \
  --override distill.teacher_config=configs/base_usef_tfgridnet.yaml \
  --override distill.teacher_checkpoint=experiments/mini_exp01_base_converged/checkpoints/best.pt \
  --override distill.teacher_weight=0.5 \
  --override distill.teacher_l1_weight=0.05 \
  --override distill.teacher_spectral_weight=0.05 \
  --override model.emb_dim=40 \
  --override model.hidden_dim=160 \
  --override model.n_heads=4 \
  --override model.n_blocks=5 \
  --override model.sparse_fusion_blocks='[0,2,4]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=5 \
  --override resume_checkpoint=experiments/mini_exp10_distill_5block_mid/checkpoints/last.pt
