#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

COMMON_OVERRIDES=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=1.5
  --override batch_size=8
  --override num_workers=0
  --override max_epochs=5
  --override model.n_fft=128
  --override model.hop_length=32
  --override model.win_length=128
  --override model.emb_dim=16
  --override model.hidden_dim=32
  --override model.n_heads=2
  --override loss.spectral_weight=0.05
)

rm -rf data/toy_matrix experiments/toy_* results/toy_* results/summary_toy_*.csv results/summary_toy_*.md

"$PY" -m cafe_tse.cli.prepare_toy_data \
  --out_dir data/toy_matrix \
  --num_samples 96 \
  --sample_rate 8000 \
  --duration 1.5 \
  --num_speakers 2

"$PY" -m cafe_tse.cli.compute_complexity_manifest \
  --manifest data/toy_matrix/toy_manifest.csv \
  --out_manifest data/toy_matrix/toy_manifest_complexity.csv \
  --sample_rate 8000 \
  --n_fft 128 \
  --hop_length 32 \
  --difficulty_rule keep

run_exp() {
  local name=$1
  local config=$2
  shift 2
  echo "=== train ${name} ==="
  "$PY" -m cafe_tse.cli.train \
    --config "$config" \
    --train_manifest data/toy_matrix/toy_manifest_complexity.csv \
    --valid_manifest data/toy_matrix/toy_manifest_complexity.csv \
    --exp_dir "experiments/${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"

  echo "=== eval ${name} ==="
  "$PY" -m cafe_tse.cli.evaluate \
    --config "$config" \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest data/toy_matrix/toy_manifest_complexity.csv \
    --out_dir "results/${name}" \
    --save_audio 3 \
    --device cuda \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

CURR='[{until_epoch: 2, difficulties: [easy]}, {until_epoch: 4, difficulties: [easy, medium]}, {until_epoch: 5, difficulties: [easy, medium, hard]}]'

run_exp toy_exp01_base configs/base_usef_tfgridnet.yaml \
  --override model.n_blocks=4 \
  --override model.sparse_fusion_blocks='[0,1,2,3]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=4

run_exp toy_exp02_lite configs/cafe_tse_lite.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=3

run_exp toy_exp03_curriculum configs/cafe_tse_curriculum.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=3 \
  --override "data.curriculum_schedule=${CURR}"

run_exp toy_exp04_dynamic configs/cafe_tse_dynamic.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=1 \
  --override model.lite_blocks=2 \
  --override model.full_blocks=3 \
  --override "data.curriculum_schedule=${CURR}"

run_exp toy_ablation_no_curriculum configs/ablation_no_curriculum.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=1 \
  --override model.lite_blocks=2 \
  --override model.full_blocks=3

run_exp toy_ablation_no_dynamic configs/ablation_no_dynamic.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=3 \
  --override "data.curriculum_schedule=${CURR}"

run_exp toy_ablation_no_sparse configs/ablation_no_sparse_fusion.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[]' \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=1 \
  --override model.lite_blocks=2 \
  --override model.full_blocks=3 \
  --override "data.curriculum_schedule=${CURR}"

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs results/toy_exp01_base results/toy_exp02_lite results/toy_exp03_curriculum results/toy_exp04_dynamic \
  --out_csv results/summary_toy_main.csv \
  --out_md results/summary_toy_main.md

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs results/toy_exp04_dynamic results/toy_ablation_no_curriculum results/toy_ablation_no_dynamic results/toy_ablation_no_sparse \
  --out_csv results/summary_toy_ablation.csv \
  --out_md results/summary_toy_ablation.md

cat results/summary_toy_main.md
printf '\n--- ablation ---\n'
cat results/summary_toy_ablation.md
