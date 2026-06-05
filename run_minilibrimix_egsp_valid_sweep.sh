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
  --override model.dynamic_inference=false
  --override model.full_blocks=5
  --override model.egsp_enabled=true
  --override model.egsp_apply_to_spec=true
  --override model.egsp_min_weight=0.80
  --override model.egsp_max_weight=1.20
)

for item in 005:0.05 010:0.10 015:0.15 020:0.20 025:0.25 030:0.30; do
  tag=${item%%:*}
  strength=${item##*:}
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$CKPT" \
    --test_manifest "$META/valid_manifest_final.csv" \
    --out_dir "results/mini_exp17_egsp_valid_s${tag}" \
    --save_audio 0 \
    --device cuda \
    "${COMMON_OVERRIDES[@]}" \
    --override model.egsp_strength="$strength"
done

"$PY" - <<'PY'
from pathlib import Path
import csv
import json

rows = []
strengths = {"005": "0.05", "010": "0.10", "015": "0.15", "020": "0.20", "025": "0.25", "030": "0.30"}
for tag in ["005", "010", "015", "020", "025", "030"]:
    name = f"mini_exp17_egsp_valid_s{tag}"
    data = json.loads(Path(f"results/{name}/summary.json").read_text(encoding="utf-8"))
    rows.append(
            {
                "method": name,
                "strength": strengths[tag],
            "si_sdri": f"{data['si_sdri']:.6f}",
            "sdr": f"{data['sdr']:.6f}",
            "sir": f"{data['sir']:.6f}",
            "sar": f"{data['sar']:.6f}",
            "rtf_eval": f"{data['rtf']:.6f}",
        }
    )
best = max(rows, key=lambda row: float(row["si_sdri"]))
Path("results/egsp_valid_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
headers = list(rows[0].keys())
with Path("results/summary_mini_egsp_valid_sweep.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
Path("results/summary_mini_egsp_valid_sweep.md").write_text("\n".join(lines), encoding="utf-8")
print(Path("results/summary_mini_egsp_valid_sweep.md").read_text(encoding="utf-8"))
print("\nBEST", best)
PY
