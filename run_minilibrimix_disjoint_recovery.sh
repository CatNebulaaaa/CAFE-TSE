#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

ROOT=data/raw/MiniLibriMix/MiniLibriMix
META=data/metadata/minilibrimix_disjoint_enroll_fixed

COMMON_DATA=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override data.normalize_audio=true
  --override data.curriculum=false
  --override max_epochs=160
  --override early_stopping.patience=40
  --override early_stopping.min_delta=0.0002
  --override optim.lr=0.001
  --override loss.si_sdr_weight=1.0
  --override loss.spectral_weight=0.1
)

EVAL_COMMON=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override data.normalize_audio=true
)

BASE_MAGMASK=(
  --override model.separator_output_mode=mag_mask
  --override model.dynamic_inference=false
  --override model.full_blocks=6
)

STUDENT_MAGMASK=(
  --override model.emb_dim=40
  --override model.hidden_dim=160
  --override model.n_heads=4
  --override model.n_blocks=5
  --override model.sparse_fusion_blocks='[0,2,4]'
  --override model.dynamic_inference=false
  --override model.full_blocks=5
  --override model.separator_output_mode=mag_mask
)

prepare_data() {
  mkdir -p "$META" results
  "$PY" -m cafe_tse.cli.prepare_librimix_manifest \
    --librimix_root "$ROOT" \
    --out_dir "$META" \
    --sample_rate 8000 \
    --num_speakers 2 \
    --mixture_type mix_clean \
    --max_train_samples 2000 \
    --max_valid_samples 150 \
    --max_test_samples 100 \
    --valid_offset 0 \
    --test_offset 100 \
    --disjoint_enrollment \
    --drop_single_enrollment_speakers

  for split in train valid test; do
    "$PY" -m cafe_tse.cli.compute_complexity_manifest \
      --manifest "$META/${split}_manifest.csv" \
      --out_manifest "$META/${split}_manifest_final.csv" \
      --sample_rate 8000 \
      --n_fft 512 \
      --hop_length 128 \
      --difficulty_rule keep
  done

  "$PY" - <<'PY'
import csv
from pathlib import Path

for split in ["train", "valid", "test"]:
    path = Path(f"data/metadata/minilibrimix_disjoint_enroll_fixed/{split}_manifest_final.csv")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    same = sum(r["target_path"] == r["enrollment_path"] for r in rows)
    print(f"{split}: n={len(rows)} target_eq_enroll={same}")
    if same:
        raise SystemExit(f"{path} still contains target/enrollment leakage")
PY
}

train_eval_bench() {
  local name="$1"
  local config="$2"
  shift 2

  echo "=== train ${name} ==="
  "$PY" -m cafe_tse.cli.train \
    --config "$config" \
    --train_manifest "$META/train_manifest_final.csv" \
    --valid_manifest "$META/valid_manifest_final.csv" \
    --exp_dir "experiments/${name}" \
    "${COMMON_DATA[@]}" \
    "$@"

  echo "=== eval ${name} ==="
  "$PY" -m cafe_tse.cli.evaluate \
    --config "$config" \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 12 \
    --device cuda \
    "${EVAL_COMMON[@]}" \
    "$@"

  echo "=== bench ${name} ==="
  "$PY" -m cafe_tse.cli.benchmark_efficiency \
    --config "$config" \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_csv "results/${name}_efficiency.csv" \
    --out_json "results/${name}_efficiency.json" \
    --device cuda \
    --num_samples 120 \
    --warmup 5 \
    "${EVAL_COMMON[@]}" \
    "$@"
}

prepare_data

train_eval_bench mini_exp42_base_magmask_disjoint_tfgridfix_lr1e3 configs/base_usef_tfgridnet.yaml "${BASE_MAGMASK[@]}"

train_eval_bench mini_exp43_student_magmask_disjoint_tfgridfix_lr1e3 configs/cafe_tse_dynamic.yaml "${STUDENT_MAGMASK[@]}"

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs \
    results/mini_exp42_base_magmask_disjoint_tfgridfix_lr1e3 \
    results/mini_exp43_student_magmask_disjoint_tfgridfix_lr1e3 \
  --out_csv results/summary_mini_disjoint_recovery_tfgridfix_lr1e3.csv \
  --out_md results/summary_mini_disjoint_recovery_tfgridfix_lr1e3.md

cat results/summary_mini_disjoint_recovery_tfgridfix_lr1e3.md
