#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
CKPT=experiments/mini_exp10_distill_5block_mid/checkpoints/best.pt

"$PY" - <<'PY'
import csv
from pathlib import Path

src = Path("data/metadata/minilibrimix_disjoint/test_manifest_final.csv")
out = Path("data/metadata/minilibrimix_disjoint/test_manifest_shuffled_enroll.csv")
rows = list(csv.DictReader(src.open(newline="", encoding="utf-8")))
paths = [row["enrollment_path"] for row in rows]
shifted = paths[1:] + paths[:1]
for row, path in zip(rows, shifted):
    row["enrollment_path"] = path
    row["enrollment_noise"] = "shuffled_mismatch"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out}")
PY

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

run_eval() {
  local name=$1
  local manifest=$2
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$CKPT" \
    --test_manifest "$manifest" \
    --out_dir "results/${name}" \
    --save_audio 0 \
    --device cuda \
    "${COMMON_OVERRIDES[@]}"
}

run_eval mini_exp19_egsp_shuffled_enroll "$META/test_manifest_shuffled_enroll.csv"
run_eval mini_exp20_egsp_enroll_1s_clean "$META/enrollment_variants/enroll_1s_clean.csv"

"$PY" - <<'PY'
from pathlib import Path
import csv
import json

methods = [
    "mini_exp10_distill_5block_mid",
    "mini_exp18_egsp_spec_s005_selected",
    "mini_exp19_egsp_shuffled_enroll",
    "mini_exp20_egsp_enroll_1s_clean",
]
rows = []
for method in methods:
    data = json.loads(Path(f"results/{method}/summary.json").read_text(encoding="utf-8"))
    rows.append(
        {
            "method": method,
            "si_sdri": f"{data['si_sdri']:.6f}",
            "sdr": f"{data['sdr']:.6f}",
            "sir": f"{data['sir']:.6f}",
            "sar": f"{data['sar']:.6f}",
            "rtf_eval": f"{data['rtf']:.6f}",
        }
    )

headers = list(rows[0].keys())
with Path("results/summary_mini_egsp_sanity.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
Path("results/summary_mini_egsp_sanity.md").write_text("\n".join(lines), encoding="utf-8")
print(Path("results/summary_mini_egsp_sanity.md").read_text(encoding="utf-8"))
PY
