#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
CKPT=experiments/mini_exp01_base_converged/checkpoints/best.pt

COMMON_OVERRIDES=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override model.dynamic_inference=true
  --override model.n_blocks=6
  --override model.full_blocks=6
  --override model.emb_dim=48
  --override model.hidden_dim=192
  --override model.n_heads=4
  --override model.sparse_fusion_blocks='[0,1,2,3,4,5]'
)

run_variant() {
  local name=$1
  shift
  echo "=== eval ${name} ==="
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/base_usef_tfgridnet.yaml \
    --checkpoint "$CKPT" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 10 \
    --device cuda \
    "${COMMON_OVERRIDES[@]}" \
    "$@"

  echo "=== bench ${name} ==="
  "$PY" -m cafe_tse.cli.benchmark_efficiency \
    --config configs/base_usef_tfgridnet.yaml \
    --checkpoint "$CKPT" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_csv "results/${name}_efficiency.csv" \
    --out_json "results/${name}_efficiency.json" \
    --device cuda \
    --num_samples 100 \
    --warmup 0 \
    "${COMMON_OVERRIDES[@]}" \
    "$@"
}

run_variant mini_exp05_cons_dyn_lite5 \
  --override model.shallow_blocks=5 \
  --override model.lite_blocks=5 \
  --override model.route_threshold_easy=0.35 \
  --override model.route_threshold_hard=0.65

run_variant mini_exp06_cons_dyn_4_5 \
  --override model.shallow_blocks=4 \
  --override model.lite_blocks=5 \
  --override model.route_threshold_easy=0.35 \
  --override model.route_threshold_hard=0.65

run_variant mini_exp07_cons_dyn_easy5 \
  --override model.shallow_blocks=5 \
  --override model.lite_blocks=6 \
  --override model.route_threshold_easy=0.55 \
  --override model.route_threshold_hard=0.75

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs \
    results/mini_exp01_base_converged \
    results/mini_exp05_cons_dyn_lite5 \
    results/mini_exp06_cons_dyn_4_5 \
    results/mini_exp07_cons_dyn_easy5 \
  --out_csv results/summary_mini_scheme_a_main.csv \
  --out_md results/summary_mini_scheme_a_main.md

"$PY" - <<'PY'
from pathlib import Path
import csv
import json

methods = [
    "mini_exp01_base_converged",
    "mini_exp05_cons_dyn_lite5",
    "mini_exp06_cons_dyn_4_5",
    "mini_exp07_cons_dyn_easy5",
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
with Path("results/summary_mini_scheme_a_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
Path("results/summary_mini_scheme_a_efficiency.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote scheme A summaries")
PY

cat results/summary_mini_scheme_a_main.md
printf '\n--- efficiency ---\n'
cat results/summary_mini_scheme_a_efficiency.md
