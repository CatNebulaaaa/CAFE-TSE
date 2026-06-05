#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
TEACHER_CKPT=experiments/mini_exp01_base_converged/checkpoints/best.pt

COMMON_OVERRIDES=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override max_epochs=60
  --override early_stopping.patience=12
  --override early_stopping.min_delta=0.0005
  --override model.n_fft=512
  --override model.hop_length=128
  --override model.win_length=512
  --override loss.spectral_weight=0.1
  --override distill.enabled=true
  --override distill.teacher_config=configs/base_usef_tfgridnet.yaml
  --override distill.teacher_checkpoint="$TEACHER_CKPT"
  --override distill.teacher_weight=0.5
  --override distill.teacher_l1_weight=0.05
  --override distill.teacher_spectral_weight=0.05
)

EVAL_OVERRIDES=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
)

run_exp() {
  local name=$1
  shift
  echo "=== train ${name} ==="
  "$PY" -m cafe_tse.cli.train \
    --config configs/cafe_tse_dynamic.yaml \
    --train_manifest "$META/train_manifest_final.csv" \
    --valid_manifest "$META/valid_manifest_final.csv" \
    --exp_dir "experiments/${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"

  echo "=== eval ${name} ==="
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 10 \
    --device cuda \
    "${EVAL_OVERRIDES[@]}" \
    "$@"

  echo "=== bench ${name} ==="
  "$PY" -m cafe_tse.cli.benchmark_efficiency \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_csv "results/${name}_efficiency.csv" \
    --out_json "results/${name}_efficiency.json" \
    --device cuda \
    --num_samples 100 \
    --warmup 0 \
    "${EVAL_OVERRIDES[@]}" \
    "$@"
}

run_exp mini_exp08_distill_4block \
  --override model.emb_dim=32 \
  --override model.hidden_dim=128 \
  --override model.n_heads=2 \
  --override model.n_blocks=4 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=false \
  --override model.full_blocks=4

run_exp mini_exp09_distill_4block_dynamic \
  --override model.emb_dim=32 \
  --override model.hidden_dim=128 \
  --override model.n_heads=2 \
  --override model.n_blocks=4 \
  --override model.sparse_fusion_blocks='[0,2]' \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=3 \
  --override model.lite_blocks=4 \
  --override model.full_blocks=4 \
  --override model.route_threshold_easy=0.45 \
  --override model.route_threshold_hard=0.70

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs \
    results/mini_exp01_base_converged \
    results/mini_exp04_dynamic_converged \
    results/mini_exp08_distill_4block \
    results/mini_exp09_distill_4block_dynamic \
  --out_csv results/summary_mini_scheme_b_main.csv \
  --out_md results/summary_mini_scheme_b_main.md

"$PY" - <<'PY'
from pathlib import Path
import csv
import json

methods = [
    "mini_exp01_base_converged",
    "mini_exp04_dynamic_converged",
    "mini_exp08_distill_4block",
    "mini_exp09_distill_4block_dynamic",
]
rows = []
for method in methods:
    path = Path(f"results/{method}_efficiency.json")
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "method": method,
        "params": data["params"],
        "rtf_wall": f"{data['rtf_wall']:.6f}",
        "active_blocks": f"{data['active_blocks']:.3f}",
        "skip_ratio": f"{data['skip_ratio']:.6f}",
        "peak_memory_mb": f"{data['peak_memory_mb']:.2f}",
        "macs_thop": "" if data["macs_thop"] is None else f"{data['macs_thop']:.0f}",
        "active_macs_proxy": "" if data["active_macs_proxy"] is None else f"{data['active_macs_proxy']:.0f}",
    })
headers = list(rows[0].keys()) if rows else ["method"]
with Path("results/summary_mini_scheme_b_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
Path("results/summary_mini_scheme_b_efficiency.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote scheme B summaries")
PY

cat results/summary_mini_scheme_b_main.md
printf '\n--- efficiency ---\n'
cat results/summary_mini_scheme_b_efficiency.md
