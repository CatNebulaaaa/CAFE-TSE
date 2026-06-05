#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

ROOT=data/raw/LibriSpeech/LibriSpeech/dev-clean
DATA=data/librispeech_tse_balanced10
META=data/metadata/librispeech_tse_balanced10

COMMON_DATA=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override data.normalize_audio=true
  --override data.curriculum=false
  --override max_epochs=100
  --override early_stopping.patience=25
  --override early_stopping.min_delta=0.0002
  --override optim.lr=0.001
  --override loss.si_sdr_weight=1.0
  --override loss.spectral_weight=0.1
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
  "$PY" -m cafe_tse.cli.prepare_librispeech_tse \
    --train_roots "$ROOT" \
    --test_roots "$ROOT" \
    --out_dir "$DATA" \
    --metadata_dir "$META" \
    --sample_rate 8000 \
    --duration 4.0 \
    --num_train 800 \
    --num_valid 150 \
    --num_test 150 \
    --min_files_per_speaker 20 \
    --max_speakers 10 \
    --shared_speaker_splits \
    --seed 123

  "$PY" -m cafe_tse.cli.compute_complexity_manifest \
    --manifest "$META/train_manifest.csv" \
    --out_manifest "$META/train_manifest_final.csv" \
    --sample_rate 8000 \
    --n_fft 512 \
    --hop_length 128 \
    --difficulty_rule keep
  "$PY" -m cafe_tse.cli.compute_complexity_manifest \
    --manifest "$META/valid_manifest.csv" \
    --out_manifest "$META/valid_manifest_final.csv" \
    --sample_rate 8000 \
    --n_fft 512 \
    --hop_length 128 \
    --difficulty_rule keep
  "$PY" -m cafe_tse.cli.compute_complexity_manifest \
    --manifest "$META/test_manifest.csv" \
    --out_manifest "$META/test_manifest_final.csv" \
    --sample_rate 8000 \
    --n_fft 512 \
    --hop_length 128 \
    --difficulty_rule keep
}

train_eval() {
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
    "${COMMON_DATA[@]}" \
    "$@"
}

if [[ "${1:-}" != "--skip-prepare" ]]; then
  prepare_data
fi

train_eval mini_exp44_balanced10_base_magmask configs/base_usef_tfgridnet.yaml "${BASE_MAGMASK[@]}"
train_eval mini_exp45_balanced10_student_magmask configs/cafe_tse_dynamic.yaml "${STUDENT_MAGMASK[@]}"

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs \
    results/mini_exp44_balanced10_base_magmask \
    results/mini_exp45_balanced10_student_magmask \
  --out_csv results/summary_librispeech_balanced10_recovery.csv \
  --out_md results/summary_librispeech_balanced10_recovery.md

cat results/summary_librispeech_balanced10_recovery.md
