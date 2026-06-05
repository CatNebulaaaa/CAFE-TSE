#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
BASE_CKPT=experiments/mini_exp01_base_converged/checkpoints/best.pt
STUDENT_CKPT=experiments/mini_exp10_distill_5block_mid/checkpoints/best.pt

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
  --override optim.lr=0.00008
)

COMMON_EVAL=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override data.curriculum=false
)

EGSP=(
  --override model.egsp_enabled=true
  --override model.egsp_strength=0.05
  --override model.egsp_min_weight=0.80
  --override model.egsp_max_weight=1.20
  --override model.egsp_apply_to_spec=true
)

STUDENT_MODEL=(
  --override model.emb_dim=40
  --override model.hidden_dim=160
  --override model.n_heads=4
  --override model.n_blocks=5
  --override model.sparse_fusion_blocks='[0,2,4]'
  --override model.dynamic_inference=false
  --override model.full_blocks=5
  "${EGSP[@]}"
)

BASE_MODEL=(
  --override model.dynamic_inference=false
  "${EGSP[@]}"
)

train_eval_student() {
  local name="$1"
  "$PY" -m cafe_tse.cli.train \
    --config configs/cafe_tse_dynamic.yaml \
    --train_manifest "$META/train_manifest_final.csv" \
    --valid_manifest "$META/valid_manifest_final.csv" \
    --exp_dir "experiments/${name}" \
    "${COMMON_TRAIN[@]}" \
    "${STUDENT_MODEL[@]}" \
    --override init_checkpoint="$STUDENT_CKPT" \
    --override init_strict=false

  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 8 \
    --device cuda \
    "${COMMON_EVAL[@]}" \
    "${STUDENT_MODEL[@]}"

  "$PY" -m cafe_tse.cli.benchmark_efficiency \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_csv "results/${name}_efficiency.csv" \
    --out_json "results/${name}_efficiency.json" \
    --device cuda \
    --num_samples 100 \
    --warmup 5 \
    "${COMMON_EVAL[@]}" \
    "${STUDENT_MODEL[@]}"
}

train_eval_base() {
  local name="$1"
  "$PY" -m cafe_tse.cli.train \
    --config configs/base_usef_tfgridnet.yaml \
    --train_manifest "$META/train_manifest_final.csv" \
    --valid_manifest "$META/valid_manifest_final.csv" \
    --exp_dir "experiments/${name}" \
    "${COMMON_TRAIN[@]}" \
    "${BASE_MODEL[@]}" \
    --override init_checkpoint="$BASE_CKPT" \
    --override init_strict=false

  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/base_usef_tfgridnet.yaml \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 8 \
    --device cuda \
    "${COMMON_EVAL[@]}" \
    "${BASE_MODEL[@]}"

  "$PY" -m cafe_tse.cli.benchmark_efficiency \
    --config configs/base_usef_tfgridnet.yaml \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_csv "results/${name}_efficiency.csv" \
    --out_json "results/${name}_efficiency.json" \
    --device cuda \
    --num_samples 100 \
    --warmup 5 \
    "${COMMON_EVAL[@]}" \
    "${BASE_MODEL[@]}"
}

echo "[1/3] 5-block Ours target fine-tune under EGSP"
train_eval_student mini_exp28_ours_egsp_target_finetune_full

echo "[2/3] 6-block quality upper-bound fine-tune under EGSP"
train_eval_base mini_exp29_base_egsp_target_finetune_full

echo "[3/3] Summarize quality boost"
"$PY" - <<'PY'
from pathlib import Path
import csv
import json

methods = [
    "mini_exp01_base_converged",
    "mini_exp10_distill_5block_mid",
    "mini_exp18_egsp_spec_s005_selected",
    "mini_exp25_baseline_egsp_quality",
    "mini_exp28_ours_egsp_target_finetune_full",
    "mini_exp29_base_egsp_target_finetune_full",
]

rows = []
for method in methods:
    summary = Path(f"results/{method}/summary.json")
    if not summary.exists():
        continue
    data = json.loads(summary.read_text(encoding="utf-8"))
    eff_path = Path(f"results/{method}_efficiency.json")
    eff = json.loads(eff_path.read_text(encoding="utf-8")) if eff_path.exists() else {}
    rows.append({
        "method": method,
        "si_sdr": f"{data.get('si_sdr', 0):.6f}",
        "si_sdri": f"{data.get('si_sdri', 0):.6f}",
        "sdr": f"{data.get('sdr', 0):.6f}",
        "sir": f"{data.get('sir', 0):.6f}",
        "sar": f"{data.get('sar', 0):.6f}",
        "pesq": f"{data.get('pesq', 0):.6f}",
        "params": data.get("params", ""),
        "rtf_wall": f"{eff.get('rtf_wall', 0):.6f}" if eff else "",
        "active_blocks": f"{eff.get('active_blocks', 0):.3f}" if eff else "",
        "macs": f"{eff.get('macs_thop', 0):.0f}" if eff and eff.get("macs_thop") is not None else "",
    })

out = Path("results/summary_mini_quality_boost.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

headers = list(rows[0].keys())
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
Path("results/summary_mini_quality_boost.md").write_text("\n".join(lines), encoding="utf-8")
print(Path("results/summary_mini_quality_boost.md").read_text(encoding="utf-8"))
PY
