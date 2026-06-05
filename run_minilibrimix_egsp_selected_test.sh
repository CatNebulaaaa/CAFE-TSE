#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
CKPT=experiments/mini_exp10_distill_5block_mid/checkpoints/best.pt
NAME=mini_exp18_egsp_spec_s005_selected

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
  --override model.dynamic_inference=false
  --override model.full_blocks=5
  --override model.egsp_enabled=true
  --override model.egsp_strength=0.05
  --override model.egsp_min_weight=0.80
  --override model.egsp_max_weight=1.20
  --override model.egsp_apply_to_spec=true
)

"$PY" -m cafe_tse.cli.evaluate \
  --config configs/cafe_tse_dynamic.yaml \
  --checkpoint "$CKPT" \
  --test_manifest "$META/test_manifest_final.csv" \
  --out_dir "results/${NAME}" \
  --save_audio 10 \
  --device cuda \
  "${COMMON_OVERRIDES[@]}"

"$PY" -m cafe_tse.cli.benchmark_efficiency \
  --config configs/cafe_tse_dynamic.yaml \
  --checkpoint "$CKPT" \
  --test_manifest "$META/test_manifest_final.csv" \
  --out_csv "results/${NAME}_efficiency.csv" \
  --out_json "results/${NAME}_efficiency.json" \
  --device cuda \
  --num_samples 100 \
  --warmup 5 \
  "${COMMON_OVERRIDES[@]}"

"$PY" - <<'PY'
from pathlib import Path
import csv
import json

methods = [
    "mini_exp01_base_converged",
    "mini_exp10_distill_5block_mid",
    "mini_exp18_egsp_spec_s005_selected",
]
rows = []
eff_rows = []
for method in methods:
    data = json.loads(Path(f"results/{method}/summary.json").read_text(encoding="utf-8"))
    rows.append(
        {
            "method": method,
            "si_sdr": f"{data['si_sdr']:.6f}",
            "si_sdri": f"{data['si_sdri']:.6f}",
            "sdr": f"{data['sdr']:.6f}",
            "sir": f"{data['sir']:.6f}",
            "sar": f"{data['sar']:.6f}",
            "rtf_eval": f"{data['rtf']:.6f}",
            "params": data["params"],
        }
    )
    eff_path = Path(f"results/{method}_efficiency.json")
    if eff_path.exists():
        eff = json.loads(eff_path.read_text(encoding="utf-8"))
        eff_rows.append(
            {
                "method": method,
                "rtf_wall": f"{eff['rtf_wall']:.6f}",
                "active_blocks": f"{eff['active_blocks']:.3f}",
                "peak_memory_mb": f"{eff['peak_memory_mb']:.2f}",
                "macs_thop": "" if eff["macs_thop"] is None else f"{eff['macs_thop']:.0f}",
            }
        )

for filename, data_rows in [
    ("results/summary_mini_egsp_selected_main.csv", rows),
    ("results/summary_mini_egsp_selected_efficiency.csv", eff_rows),
]:
    headers = list(data_rows[0].keys()) if data_rows else ["method"]
    with Path(filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data_rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in data_rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    Path(filename.replace(".csv", ".md")).write_text("\n".join(lines), encoding="utf-8")

print(Path("results/summary_mini_egsp_selected_main.md").read_text(encoding="utf-8"))
print("\n--- efficiency ---")
print(Path("results/summary_mini_egsp_selected_efficiency.md").read_text(encoding="utf-8"))
PY
