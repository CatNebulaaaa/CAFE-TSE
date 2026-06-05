#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export LD_LIBRARY_PATH=/root/miniconda3/envs/cafe-tse/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=src:third_party/asteroid_site:third_party/speakerbeam/src

WAIT_PID="${1:-}"
TRAIN=data/metadata/librispeech_tse_shared_clean80/train_manifest_final.csv
VALID=data/metadata/librispeech_tse_shared_clean80/valid_manifest_final.csv
TEST=data/metadata/librispeech_tse_shared_clean80/test_manifest_final.csv

MID_EXP=experiments/open_speakerbeam_shared_clean80_mid
STRONG_EXP=experiments/open_speakerbeam_shared_clean80_strong
STUDENT_EXP=experiments/open_speakerbeam_shared_clean80_student_small
DISTILL_EXP=experiments/open_speakerbeam_shared_clean80_student_small_distill

if [[ -n "$WAIT_PID" ]]; then
  echo "waiting for strong teacher pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 300
  done
fi

if [[ ! -f "$MID_EXP/best.pt" ]]; then
  echo "missing baseline teacher checkpoint: $MID_EXP/best.pt" >&2
  exit 1
fi

TEACHER_EXP="$MID_EXP"
if [[ -f "$STRONG_EXP/summary.json" ]]; then
  TEACHER_EXP="$("$PY" - "$MID_EXP/summary.json" "$STRONG_EXP/summary.json" "$MID_EXP" "$STRONG_EXP" <<'PY'
import json
import sys

mid_summary, strong_summary, mid_exp, strong_exp = sys.argv[1:5]
with open(mid_summary, encoding="utf-8") as f:
    mid = json.load(f)
with open(strong_summary, encoding="utf-8") as f:
    strong = json.load(f)
mid_score = float(mid.get("si_sdr", float("-inf")))
strong_score = float(strong.get("si_sdr", float("-inf")))
print(strong_exp if strong_score > mid_score else mid_exp)
PY
)"
fi
echo "selected_teacher=$TEACHER_EXP"

if [[ ! -f "$STUDENT_EXP/summary.json" ]]; then
  "$PY" scripts/train_open_speakerbeam.py \
    --train_manifest "$TRAIN" \
    --valid_manifest "$VALID" \
    --test_manifest "$TEST" \
    --exp_dir "$STUDENT_EXP" \
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
else
  echo "student summary exists, skipping supervised student: $STUDENT_EXP/summary.json"
fi

if [[ ! -f "$DISTILL_EXP/summary.json" ]]; then
  "$PY" scripts/train_open_speakerbeam_distill.py \
    --teacher_checkpoint "$TEACHER_EXP/best.pt" \
    --train_manifest "$TRAIN" \
    --valid_manifest "$VALID" \
    --test_manifest "$TEST" \
    --exp_dir "$DISTILL_EXP" \
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
else
  echo "distill summary exists, skipping distilled student: $DISTILL_EXP/summary.json"
fi

"$PY" - <<'PY'
import json
from pathlib import Path

names = {
    "mid_teacher": Path("experiments/open_speakerbeam_shared_clean80_mid/summary.json"),
    "strong_teacher": Path("experiments/open_speakerbeam_shared_clean80_strong/summary.json"),
    "student_small": Path("experiments/open_speakerbeam_shared_clean80_student_small/summary.json"),
    "student_distill": Path("experiments/open_speakerbeam_shared_clean80_student_small_distill/summary.json"),
}
for name, path in names.items():
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(name, "si_sdr", data.get("si_sdr"), "sdr", data.get("sdr"), "sir", data.get("sir"))
PY
