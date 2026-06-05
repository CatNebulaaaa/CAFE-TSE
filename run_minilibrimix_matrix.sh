#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

ROOT=data/raw/MiniLibriMix/MiniLibriMix
META=data/metadata/minilibrimix

COMMON_OVERRIDES=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override max_epochs=8
  --override model.n_fft=256
  --override model.hop_length=64
  --override model.win_length=256
  --override model.emb_dim=24
  --override model.hidden_dim=64
  --override model.n_heads=2
  --override loss.spectral_weight=0.05
)

rm -rf "$META" experiments/mini_* results/mini_* results/summary_mini_*.csv results/summary_mini_*.md

"$PY" -m cafe_tse.cli.prepare_librimix_manifest \
  --librimix_root "$ROOT" \
  --out_dir "$META" \
  --sample_rate 8000 \
  --num_speakers 2 \
  --mixture_type mix_clean \
  --max_train_samples 800 \
  --max_valid_samples 100 \
  --max_test_samples 200

for split in train valid test; do
  "$PY" -m cafe_tse.cli.compute_complexity_manifest \
    --manifest "$META/${split}_manifest.csv" \
    --out_manifest "$META/${split}_manifest_final.csv" \
    --sample_rate 8000 \
    --n_fft 256 \
    --hop_length 64 \
    --difficulty_rule keep
done

run_exp() {
  local name=$1
  local config=$2
  shift 2
  echo "=== train ${name} ==="
  "$PY" -m cafe_tse.cli.train \
    --config "$config" \
    --train_manifest "$META/train_manifest_final.csv" \
    --valid_manifest "$META/valid_manifest_final.csv" \
    --exp_dir "experiments/${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"

  echo "=== eval ${name} ==="
  "$PY" -m cafe_tse.cli.evaluate \
    --config "$config" \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 10 \
    --device cuda \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

CURR='[{until_epoch: 3, difficulties: [easy]}, {until_epoch: 6, difficulties: [easy, medium]}, {until_epoch: 8, difficulties: [easy, medium, hard]}]'

run_exp mini_exp01_base configs/base_usef_tfgridnet.yaml \
  --override model.n_blocks=4 \
  --override model.sparse_fusion_blocks='[0,1,2,3]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=4

run_exp mini_exp02_lite configs/cafe_tse_lite.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=3

run_exp mini_exp03_curriculum configs/cafe_tse_curriculum.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=3 \
  --override "data.curriculum_schedule=${CURR}"

run_exp mini_exp04_dynamic configs/cafe_tse_dynamic.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=1 \
  --override model.lite_blocks=2 \
  --override model.full_blocks=3 \
  --override model.route_threshold_easy=0.45 \
  --override model.route_threshold_hard=0.70 \
  --override "data.curriculum_schedule=${CURR}"

run_exp mini_ablation_no_curriculum configs/ablation_no_curriculum.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=1 \
  --override model.lite_blocks=2 \
  --override model.full_blocks=3 \
  --override model.route_threshold_easy=0.45 \
  --override model.route_threshold_hard=0.70

run_exp mini_ablation_no_dynamic configs/ablation_no_dynamic.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=3 \
  --override "data.curriculum_schedule=${CURR}"

run_exp mini_ablation_no_sparse configs/ablation_no_sparse_fusion.yaml \
  --override model.n_blocks=3 \
  --override model.sparse_fusion_blocks='[]' \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=1 \
  --override model.lite_blocks=2 \
  --override model.full_blocks=3 \
  --override model.route_threshold_easy=0.45 \
  --override model.route_threshold_hard=0.70 \
  --override "data.curriculum_schedule=${CURR}"

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs results/mini_exp01_base results/mini_exp02_lite results/mini_exp03_curriculum results/mini_exp04_dynamic \
  --out_csv results/summary_mini_main.csv \
  --out_md results/summary_mini_main.md

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs results/mini_exp04_dynamic results/mini_ablation_no_curriculum results/mini_ablation_no_dynamic results/mini_ablation_no_sparse \
  --out_csv results/summary_mini_ablation.csv \
  --out_md results/summary_mini_ablation.md

"$PY" - <<'PY'
from pathlib import Path
import pandas as pd

df = pd.read_csv("results/mini_exp04_dynamic/metrics.csv")
g = df.groupby("difficulty").agg(
    si_sdri=("si_sdri", "mean"),
    sdr=("sdr", "mean"),
    sir=("sir", "mean"),
    sar=("sar", "mean"),
    rtf=("rtf", "mean"),
    skip_ratio=("skip_ratio", "mean"),
    active_blocks=("active_blocks", "mean"),
    n=("utt_id", "count"),
).reset_index()
g.to_csv("results/summary_mini_difficulty.csv", index=False)
headers = list(g.columns)
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for _, row in g.iterrows():
    vals = []
    for h in headers:
        v = row[h]
        vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
    lines.append("| " + " | ".join(vals) + " |")
Path("results/summary_mini_difficulty.md").write_text("\n".join(lines), encoding="utf-8")
PY

cat results/summary_mini_main.md
printf '\n--- ablation ---\n'
cat results/summary_mini_ablation.md
printf '\n--- difficulty ---\n'
cat results/summary_mini_difficulty.md
