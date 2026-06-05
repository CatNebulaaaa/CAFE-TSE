#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
BASE_CKPT=experiments/mini_exp01_base_converged/checkpoints/best.pt
STUDENT_CKPT=experiments/mini_exp10_distill_5block_mid/checkpoints/best.pt

COMMON_MODEL=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=6
  --override num_workers=0
  --override model.emb_dim=40
  --override model.hidden_dim=160
  --override model.n_heads=4
  --override model.n_blocks=5
  --override model.sparse_fusion_blocks='[0,2,4]'
  --override model.dynamic_inference=false
  --override model.full_blocks=5
  --override model.egsp_enabled=true
  --override model.egsp_strength=0.05
  --override model.egsp_min_weight=0.80
  --override model.egsp_max_weight=1.20
  --override model.egsp_apply_to_spec=true
)

DISTILL=(
  --override distill.enabled=true
  --override distill.teacher_config=configs/base_usef_tfgridnet.yaml
  --override distill.teacher_checkpoint="$BASE_CKPT"
  --override distill.teacher_weight=0.50
  --override distill.teacher_l1_weight=0.05
  --override distill.teacher_spectral_weight=0.05
)

train_exp() {
  local name="$1"
  shift
  "$PY" -m cafe_tse.cli.train \
    --config configs/cafe_tse_dynamic.yaml \
    --train_manifest "$META/train_manifest_final.csv" \
    --valid_manifest "$META/valid_manifest_final.csv" \
    --exp_dir "experiments/${name}" \
    "${COMMON_MODEL[@]}" \
    --override init_checkpoint="$STUDENT_CKPT" \
    --override init_strict=false \
    --override optim.lr=0.00008 \
    --override max_epochs=24 \
    --override early_stopping.patience=6 \
    --override early_stopping.min_delta=0.0002 \
    "${DISTILL[@]}" \
    "$@"
}

eval_exp() {
  local name="$1"
  local ckpt="$2"
  shift 2
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$ckpt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 8 \
    --device cuda \
    "${COMMON_MODEL[@]}" \
    "$@"

  "$PY" -m cafe_tse.cli.benchmark_efficiency \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$ckpt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_csv "results/${name}_efficiency.csv" \
    --out_json "results/${name}_efficiency.json" \
    --device cuda \
    --num_samples 100 \
    --warmup 5 \
    "${COMMON_MODEL[@]}" \
    "$@"
}

echo "[1/5] Dynamic sparse fusion eval on converged student"
eval_exp mini_exp21_dynamic_sparse_fusion_eval "$STUDENT_CKPT" \
  --override model.dynamic_sparse_fusion=true

echo "[2/5] Gated conditioning + multi-resolution spectral fine-tune"
train_exp mini_exp22_gated_mrstft_finetune \
  --override model.condition_fusion_mode=gated \
  --override loss.multi_resolution_spectral_weight=0.10 \
  --override loss.multi_resolution_fft_sizes='[256,512,1024]'
eval_exp mini_exp22_gated_mrstft_finetune experiments/mini_exp22_gated_mrstft_finetune/checkpoints/best.pt \
  --override model.condition_fusion_mode=gated

echo "[3/5] Depth-aware curriculum distillation fine-tune"
train_exp mini_exp23_depthaware_finetune \
  --override depth_aware.enabled=true \
  --override depth_aware.active_depths='[3,4,5]' \
  --override depth_aware.probabilities='[0.20,0.30,0.50]'
eval_exp mini_exp23_depthaware_static experiments/mini_exp23_depthaware_finetune/checkpoints/best.pt
eval_exp mini_exp23_depthaware_dynamic experiments/mini_exp23_depthaware_finetune/checkpoints/best.pt \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=3 \
  --override model.lite_blocks=4 \
  --override model.full_blocks=5 \
  --override model.route_threshold_easy=0.53 \
  --override model.route_threshold_hard=0.57

echo "[4/5] Combined system fine-tune"
train_exp mini_exp24_ours_system_full \
  --override model.condition_fusion_mode=gated \
  --override model.dynamic_sparse_fusion=true \
  --override loss.multi_resolution_spectral_weight=0.10 \
  --override loss.multi_resolution_fft_sizes='[256,512,1024]' \
  --override depth_aware.enabled=true \
  --override depth_aware.active_depths='[3,4,5]' \
  --override depth_aware.probabilities='[0.20,0.30,0.50]'
eval_exp mini_exp24_ours_system_full_static experiments/mini_exp24_ours_system_full/checkpoints/best.pt \
  --override model.condition_fusion_mode=gated \
  --override model.dynamic_sparse_fusion=true
eval_exp mini_exp24_ours_system_full_dynamic experiments/mini_exp24_ours_system_full/checkpoints/best.pt \
  --override model.condition_fusion_mode=gated \
  --override model.dynamic_sparse_fusion=true \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=3 \
  --override model.lite_blocks=4 \
  --override model.full_blocks=5 \
  --override model.route_threshold_easy=0.53 \
  --override model.route_threshold_hard=0.57

echo "[5/5] Summarize innovation experiments"
"$PY" - <<'PY'
from pathlib import Path
import csv
import json

methods = [
    "mini_exp01_base_converged",
    "mini_exp10_distill_5block_mid",
    "mini_exp18_egsp_spec_s005_selected",
    "mini_exp21_dynamic_sparse_fusion_eval",
    "mini_exp22_gated_mrstft_finetune",
    "mini_exp23_depthaware_static",
    "mini_exp23_depthaware_dynamic",
    "mini_exp24_ours_system_full_static",
    "mini_exp24_ours_system_full_dynamic",
]

rows = []
for method in methods:
    summary_path = Path(f"results/{method}/summary.json")
    if not summary_path.exists():
        continue
    m = json.loads(summary_path.read_text(encoding="utf-8"))
    eff_path = Path(f"results/{method}_efficiency.json")
    eff = json.loads(eff_path.read_text(encoding="utf-8")) if eff_path.exists() else {}
    rows.append({
        "method": method,
        "si_sdr": f"{m.get('si_sdr', 0):.6f}",
        "si_sdri": f"{m.get('si_sdri', 0):.6f}",
        "sdr": f"{m.get('sdr', 0):.6f}",
        "sir": f"{m.get('sir', 0):.6f}",
        "sar": f"{m.get('sar', 0):.6f}",
        "rtf_eval": f"{m.get('rtf', 0):.6f}",
        "params": m.get("params", ""),
        "rtf_wall": f"{eff.get('rtf_wall', 0):.6f}" if eff else "",
        "active_blocks": f"{eff.get('active_blocks', 0):.3f}" if eff else "",
        "macs": f"{eff.get('macs_thop', 0):.0f}" if eff and eff.get("macs_thop") else "",
        "peak_memory_mb": f"{eff.get('peak_memory_mb', 0):.2f}" if eff else "",
    })

out = Path("results/summary_mini_system_innovations.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

headers = list(rows[0].keys())
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
Path("results/summary_mini_system_innovations.md").write_text("\n".join(lines), encoding="utf-8")
print(Path("results/summary_mini_system_innovations.md").read_text(encoding="utf-8"))
PY
