#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
STUDENT_CKPT=experiments/mini_exp10_distill_5block_mid/checkpoints/best.pt
NAME=mini_exp30_ours_egsp_magmask_finetune

COMMON_TRAIN=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=6
  --override num_workers=0
  --override data.curriculum=false
  --override max_epochs=48
  --override early_stopping.patience=10
  --override early_stopping.min_delta=0.0002
  --override optim.lr=0.0001
  --override loss.si_sdr_weight=1.0
  --override loss.spectral_weight=0.5
)

COMMON_EVAL=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override data.curriculum=false
)

MODEL=(
  --override model.emb_dim=40
  --override model.hidden_dim=160
  --override model.n_heads=4
  --override model.n_blocks=5
  --override model.sparse_fusion_blocks='[0,2,4]'
  --override model.dynamic_inference=false
  --override model.full_blocks=5
  --override model.separator_output_mode=mag_mask
  --override model.egsp_enabled=true
  --override model.egsp_strength=0.05
  --override model.egsp_min_weight=0.80
  --override model.egsp_max_weight=1.20
  --override model.egsp_apply_to_spec=true
)

echo "[1/3] train ${NAME}"
"$PY" -m cafe_tse.cli.train \
  --config configs/cafe_tse_dynamic.yaml \
  --train_manifest "$META/train_manifest_final.csv" \
  --valid_manifest "$META/valid_manifest_final.csv" \
  --exp_dir "experiments/${NAME}" \
  "${COMMON_TRAIN[@]}" \
  "${MODEL[@]}" \
  --override init_checkpoint="$STUDENT_CKPT" \
  --override init_strict=false

echo "[2/3] evaluate ${NAME}"
"$PY" -m cafe_tse.cli.evaluate \
  --config configs/cafe_tse_dynamic.yaml \
  --checkpoint "experiments/${NAME}/checkpoints/best.pt" \
  --test_manifest "$META/test_manifest_final.csv" \
  --out_dir "results/${NAME}" \
  --save_audio 8 \
  --device cuda \
  "${COMMON_EVAL[@]}" \
  "${MODEL[@]}"

echo "[3/3] summary"
cat "results/${NAME}/summary.json"
