#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/root/miniconda3/envs/cafe-tse/lib
export PYTHONPATH=/root/autodl-tmp/CAFE-TSE/src

cd /root/autodl-tmp/CAFE-TSE
mkdir -p logs results/stable_small_improvements

PY=/root/miniconda3/envs/cafe-tse/bin/python
TRAIN=data/metadata/librispeech_tse_shared_clean80/train_manifest_final.csv
VALID=data/metadata/librispeech_tse_shared_clean80/valid_manifest_final.csv
TEST=data/metadata/librispeech_tse_shared_clean80/test_manifest_final.csv
MID_FT=experiments/open_speakerbeam_shared_clean80_student_mid_distill_ft/best.pt

echo "[1/3] embedding-level multi-enrollment pooling"
$PY scripts/evaluate_open_speakerbeam_embedding_pool.py \
  --checkpoint "$MID_FT" \
  --test_manifest "$TEST" \
  --pool_manifest "$TRAIN" \
  --out_dir results/stable_small_improvements/embedding_pool_mid_ft \
  --ks 1,2,4,8 \
  --device cuda

echo "[2/3] training-time EGSP fine-tune, strength=0.02"
$PY scripts/train_open_speakerbeam_egsp.py \
  --train_manifest "$TRAIN" \
  --valid_manifest "$VALID" \
  --test_manifest "$TEST" \
  --exp_dir experiments/open_speakerbeam_shared_clean80_student_mid_egsp_train_s002 \
  --init_checkpoint "$MID_FT" \
  --egsp_strength 0.02 \
  --egsp_min_weight 0.90 \
  --egsp_max_weight 1.10 \
  --lr 0.0001 \
  --epochs 18 \
  --batch_size 4 \
  --n_filters 256 --bn_chan 64 --hid_chan 256 --skip_chan 64 \
  --adapt_enroll_dim 64 --n_blocks 6 --n_repeats 2 --i_adapt_layer 5 \
  --device cuda

echo "[3/3] training-time EGSP fine-tune, strength=0.05"
$PY scripts/train_open_speakerbeam_egsp.py \
  --train_manifest "$TRAIN" \
  --valid_manifest "$VALID" \
  --test_manifest "$TEST" \
  --exp_dir experiments/open_speakerbeam_shared_clean80_student_mid_egsp_train_s005 \
  --init_checkpoint "$MID_FT" \
  --egsp_strength 0.05 \
  --egsp_min_weight 0.85 \
  --egsp_max_weight 1.15 \
  --lr 0.0001 \
  --epochs 18 \
  --batch_size 4 \
  --n_filters 256 --bn_chan 64 --hid_chan 256 --skip_chan 64 \
  --adapt_enroll_dim 64 --n_blocks 6 --n_repeats 2 --i_adapt_layer 5 \
  --device cuda

echo "[done] summaries"
$PY - <<'PY'
from pathlib import Path
paths = [
    Path("results/stable_small_improvements/embedding_pool_mid_ft/best_embedding_pool.json"),
    Path("experiments/open_speakerbeam_shared_clean80_student_mid_egsp_train_s002/summary.json"),
    Path("experiments/open_speakerbeam_shared_clean80_student_mid_egsp_train_s005/summary.json"),
]
for p in paths:
    print("===", p)
    print(p.read_text() if p.exists() else "missing")
PY
