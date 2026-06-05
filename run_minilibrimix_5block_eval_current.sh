#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
CKPT=experiments/mini_exp10_distill_5block_mid/checkpoints/best.pt

COMMON_OVERRIDES=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override model.emb_dim=40
  --override model.hidden_dim=160
  --override model.n_heads=4
  --override model.n_blocks=5
  --override model.sparse_fusion_blocks='[0,2,4]'
)

run_eval_bench() {
  local name=$1
  shift
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$CKPT" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 10 \
    --device cuda \
    "${COMMON_OVERRIDES[@]}" \
    "$@"

  "$PY" -m cafe_tse.cli.benchmark_efficiency \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$CKPT" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_csv "results/${name}_efficiency.csv" \
    --out_json "results/${name}_efficiency.json" \
    --device cuda \
    --num_samples 100 \
    --warmup 5 \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

run_eval_bench mini_exp10_distill_5block_mid \
  --override model.dynamic_inference=false \
  --override model.full_blocks=5

run_eval_bench mini_exp11_distill_5block_mid_dynamic \
  --override model.dynamic_inference=true \
  --override model.shallow_blocks=4 \
  --override model.lite_blocks=5 \
  --override model.full_blocks=5 \
  --override model.route_threshold_easy=0.45 \
  --override model.route_threshold_hard=0.70

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs \
    results/mini_exp01_base_converged \
    results/mini_exp04_dynamic_converged \
    results/mini_exp08_distill_4block \
    results/mini_exp10_distill_5block_mid \
    results/mini_exp11_distill_5block_mid_dynamic \
  --out_csv results/summary_mini_distill_5block_main.csv \
  --out_md results/summary_mini_distill_5block_main.md

"$PY" - <<'PY'
from pathlib import Path
import csv
import json

methods = [
    "mini_exp01_base_converged",
    "mini_exp04_dynamic_converged",
    "mini_exp08_distill_4block",
    "mini_exp10_distill_5block_mid",
    "mini_exp11_distill_5block_mid_dynamic",
]
rows = []
for method in methods:
    path = Path(f"results/{method}_efficiency.json")
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "method": method,
            "params": data["params"],
            "rtf_wall": f"{data['rtf_wall']:.6f}",
            "active_blocks": f"{data['active_blocks']:.3f}",
            "skip_ratio": f"{data['skip_ratio']:.6f}",
            "peak_memory_mb": f"{data['peak_memory_mb']:.2f}",
            "macs_thop": "" if data["macs_thop"] is None else f"{data['macs_thop']:.0f}",
            "active_macs_proxy": "" if data["active_macs_proxy"] is None else f"{data['active_macs_proxy']:.0f}",
        }
    )

headers = list(rows[0].keys()) if rows else ["method"]
with Path("results/summary_mini_distill_5block_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
Path("results/summary_mini_distill_5block_efficiency.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote 5-block distillation summaries")
PY

cat results/summary_mini_distill_5block_main.md
printf '\n--- efficiency ---\n'
cat results/summary_mini_distill_5block_efficiency.md
