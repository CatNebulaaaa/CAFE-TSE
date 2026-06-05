#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/cafe-tse/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src:third_party/asteroid_site:third_party/speakerbeam/src

COMMON_MODEL=(
  --batch_size 2
  --lr 0.001
  --n_filters 512
  --bn_chan 128
  --hid_chan 512
  --skip_chan 128
  --adapt_enroll_dim 128
  --n_blocks 8
  --n_repeats 3
  --i_adapt_layer 7
)

train_balanced10_strong() {
  "$PY" scripts/train_open_speakerbeam.py \
    --train_manifest data/metadata/librispeech_tse_balanced10/train_manifest_final.csv \
    --valid_manifest data/metadata/librispeech_tse_balanced10/valid_manifest_final.csv \
    --test_manifest data/metadata/librispeech_tse_balanced10/test_manifest_final.csv \
    --exp_dir experiments/open_speakerbeam_balanced10_strong \
    --epochs 80 \
    "${COMMON_MODEL[@]}"
}

train_balanced40_strong() {
  "$PY" scripts/train_open_speakerbeam.py \
    --train_manifest data/metadata/librispeech_tse_balanced40/train_manifest_final.csv \
    --valid_manifest data/metadata/librispeech_tse_balanced40/valid_manifest_final.csv \
    --test_manifest data/metadata/librispeech_tse_balanced40/test_manifest_final.csv \
    --exp_dir experiments/open_speakerbeam_balanced40_strong \
    --epochs 80 \
    "${COMMON_MODEL[@]}"
}

continue_balanced10_mid() {
  "$PY" scripts/train_open_speakerbeam.py \
    --train_manifest data/metadata/librispeech_tse_balanced10/train_manifest_final.csv \
    --valid_manifest data/metadata/librispeech_tse_balanced10/valid_manifest_final.csv \
    --test_manifest data/metadata/librispeech_tse_balanced10/test_manifest_final.csv \
    --exp_dir experiments/open_speakerbeam_balanced10_mid \
    --resume_checkpoint experiments/open_speakerbeam_balanced10_mid/best.pt \
    --epochs 100 \
    --batch_size 4 \
    --lr 0.0003 \
    --n_filters 256 \
    --bn_chan 64 \
    --hid_chan 256 \
    --skip_chan 64 \
    --adapt_enroll_dim 64 \
    --n_blocks 6 \
    --n_repeats 2 \
    --i_adapt_layer 5
}

train_shared_clean80_strong() {
  "$PY" scripts/train_open_speakerbeam.py \
    --train_manifest data/metadata/librispeech_tse_shared_clean80/train_manifest_final.csv \
    --valid_manifest data/metadata/librispeech_tse_shared_clean80/valid_manifest_final.csv \
    --test_manifest data/metadata/librispeech_tse_shared_clean80/test_manifest_final.csv \
    --exp_dir experiments/open_speakerbeam_shared_clean80_strong \
    --epochs 100 \
    "${COMMON_MODEL[@]}"
}

train_shared_clean80_student_small() {
  "$PY" scripts/train_open_speakerbeam.py \
    --train_manifest data/metadata/librispeech_tse_shared_clean80/train_manifest_final.csv \
    --valid_manifest data/metadata/librispeech_tse_shared_clean80/valid_manifest_final.csv \
    --test_manifest data/metadata/librispeech_tse_shared_clean80/test_manifest_final.csv \
    --exp_dir experiments/open_speakerbeam_shared_clean80_student_small \
    --epochs 80 \
    --batch_size 6 \
    --lr 0.001 \
    --n_filters 128 \
    --bn_chan 32 \
    --hid_chan 128 \
    --skip_chan 32 \
    --adapt_enroll_dim 32 \
    --n_blocks 4 \
    --n_repeats 2 \
    --i_adapt_layer 3
}

distill_shared_clean80_student_small() {
  "$PY" scripts/train_open_speakerbeam_distill.py \
    --teacher_checkpoint experiments/open_speakerbeam_shared_clean80_mid/best.pt \
    --train_manifest data/metadata/librispeech_tse_shared_clean80/train_manifest_final.csv \
    --valid_manifest data/metadata/librispeech_tse_shared_clean80/valid_manifest_final.csv \
    --test_manifest data/metadata/librispeech_tse_shared_clean80/test_manifest_final.csv \
    --exp_dir experiments/open_speakerbeam_shared_clean80_student_small_distill \
    --epochs 80 \
    --batch_size 6 \
    --lr 0.001 \
    --target_weight 1.0 \
    --teacher_weight 0.5 \
    --n_filters 128 \
    --bn_chan 32 \
    --hid_chan 128 \
    --skip_chan 32 \
    --adapt_enroll_dim 32 \
    --n_blocks 4 \
    --n_repeats 2 \
    --i_adapt_layer 3
}

case "${1:-balanced10}" in
  balanced10)
    train_balanced10_strong
    ;;
  balanced40)
    train_balanced40_strong
    ;;
  both)
    train_balanced10_strong
    train_balanced40_strong
    ;;
  continue-balanced10-mid)
    continue_balanced10_mid
    ;;
  shared-clean80-strong)
    train_shared_clean80_strong
    ;;
  shared-clean80-student-small)
    train_shared_clean80_student_small
    ;;
  shared-clean80-distill-small)
    distill_shared_clean80_student_small
    ;;
  *)
    echo "usage: $0 [balanced10|balanced40|both|continue-balanced10-mid|shared-clean80-strong|shared-clean80-student-small|shared-clean80-distill-small]" >&2
    exit 2
    ;;
esac
