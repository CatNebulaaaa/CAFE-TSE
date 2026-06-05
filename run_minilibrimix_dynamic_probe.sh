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
  --override model.dynamic_inference=true
)

run_probe() {
  local name=$1
  shift
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$CKPT" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 0 \
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

run_probe mini_exp12_distill_5block_dyn_lite4 \
  --override model.shallow_blocks=4 \
  --override model.lite_blocks=4 \
  --override model.full_blocks=5 \
  --override model.route_threshold_easy=0.45 \
  --override model.route_threshold_hard=0.70

run_probe mini_exp13_distill_5block_dyn_mixed345 \
  --override model.shallow_blocks=3 \
  --override model.lite_blocks=4 \
  --override model.full_blocks=5 \
  --override model.route_threshold_easy=0.54 \
  --override model.route_threshold_hard=0.59

"$PY" - <<'PY'
from pathlib import Path
import csv
import json

methods = [
    "mini_exp01_base_converged",
    "mini_exp10_distill_5block_mid",
    "mini_exp11_distill_5block_mid_dynamic",
    "mini_exp12_distill_5block_dyn_lite4",
    "mini_exp13_distill_5block_dyn_mixed345",
]
main_rows = []
for method in methods:
    summary = Path(f"results/{method}/summary.json")
    if not summary.exists():
        continue
    data = json.loads(summary.read_text(encoding="utf-8"))
    main_rows.append(
        {
            "method": method,
            "si_sdri": f"{data['si_sdri']:.6f}",
            "sdr": f"{data['sdr']:.6f}",
            "sir": f"{data['sir']:.6f}",
            "sar": f"{data['sar']:.6f}",
            "rtf_eval": f"{data['rtf']:.6f}",
            "skip_ratio_eval": f"{data['skip_ratio']:.6f}",
            "params": data["params"],
        }
    )
eff_rows = []
for method in methods:
    path = Path(f"results/{method}_efficiency.json")
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    eff_rows.append(
        {
            "method": method,
            "rtf_wall": f"{data['rtf_wall']:.6f}",
            "active_blocks": f"{data['active_blocks']:.3f}",
            "skip_ratio": f"{data['skip_ratio']:.6f}",
            "peak_memory_mb": f"{data['peak_memory_mb']:.2f}",
            "macs_thop": "" if data["macs_thop"] is None else f"{data['macs_thop']:.0f}",
            "active_macs_proxy": "" if data["active_macs_proxy"] is None else f"{data['active_macs_proxy']:.0f}",
        }
    )

for filename, rows in [
    ("results/summary_mini_dynamic_probe_main.csv", main_rows),
    ("results/summary_mini_dynamic_probe_efficiency.csv", eff_rows),
]:
    headers = list(rows[0].keys()) if rows else ["method"]
    with Path(filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        md.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    Path(filename.replace(".csv", ".md")).write_text("\n".join(md), encoding="utf-8")

print(Path("results/summary_mini_dynamic_probe_main.md").read_text(encoding="utf-8"))
print("\n--- efficiency ---")
print(Path("results/summary_mini_dynamic_probe_efficiency.md").read_text(encoding="utf-8"))
PY
