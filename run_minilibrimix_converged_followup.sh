#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

ROOT=data/raw/MiniLibriMix/MiniLibriMix
META=data/metadata/minilibrimix_disjoint

COMMON_OVERRIDES=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
  --override max_epochs=60
  --override early_stopping.patience=12
  --override early_stopping.min_delta=0.0005
)

EVAL_OVERRIDES=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
)

mkdir -p "$META" results

"$PY" -m cafe_tse.cli.prepare_librimix_manifest \
  --librimix_root "$ROOT" \
  --out_dir "$META" \
  --sample_rate 8000 \
  --num_speakers 2 \
  --mixture_type mix_clean \
  --max_train_samples 800 \
  --max_valid_samples 100 \
  --max_test_samples 100 \
  --valid_offset 0 \
  --test_offset 100

for split in train valid test; do
  "$PY" -m cafe_tse.cli.compute_complexity_manifest \
    --manifest "$META/${split}_manifest.csv" \
    --out_manifest "$META/${split}_manifest_final.csv" \
    --sample_rate 8000 \
    --n_fft 512 \
    --hop_length 128 \
    --difficulty_rule keep
done

train_eval() {
  local name=$1
  local config=$2
  shift 2
  echo "=== train ${name} ==="
  "$PY" -m cafe_tse.cli.train \
    --config "$config" \
    --train_manifest "$META/train_manifest_final.csv" \
    --valid_manifest "$META/valid_manifest_final.csv" \
    --exp_dir "experiments/${name}" \
    "${COMMON_OVERRIDES[@]}" \
    "$@"

  echo "=== eval ${name} ==="
  "$PY" -m cafe_tse.cli.evaluate \
    --config "$config" \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio 10 \
    --device cuda \
    "${EVAL_OVERRIDES[@]}" \
    "$@"
}

train_eval mini_exp01_base_converged configs/base_usef_tfgridnet.yaml
train_eval mini_exp04_dynamic_converged configs/cafe_tse_dynamic.yaml

"$PY" -m cafe_tse.cli.summarize_results \
  --result_dirs results/mini_exp01_base_converged results/mini_exp04_dynamic_converged \
  --out_csv results/summary_mini_converged_main.csv \
  --out_md results/summary_mini_converged_main.md

"$PY" -m cafe_tse.cli.prepare_enrollment_variants \
  --manifest "$META/test_manifest_final.csv" \
  --out_dir "$META/enrollment_variants" \
  --sample_rate 8000 \
  --durations 1 3 5 \
  --noise_snr_db 10

while IFS=, read -r variant manifest duration noise; do
  if [[ "$variant" == "variant" ]]; then
    continue
  fi
  echo "=== eval enrollment ${variant} ==="
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint experiments/mini_exp04_dynamic_converged/checkpoints/best.pt \
    --test_manifest "$manifest" \
    --out_dir "results/enrollment_${variant}" \
    --save_audio 0 \
    --device cuda \
    "${EVAL_OVERRIDES[@]}"
done < "$META/enrollment_variants/variants.csv"

"$PY" - <<'PY'
from pathlib import Path
import csv

rows = []
for path in sorted(Path("results").glob("enrollment_*/metrics.csv")):
    data = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not data:
        continue
    variant = path.parent.name.replace("enrollment_", "")
    def mean(key):
        vals = [float(r[key]) for r in data if r.get(key) not in ("", "nan", None)]
        return sum(vals) / max(len(vals), 1)
    rows.append({
        "variant": variant,
        "si_sdri": mean("si_sdri"),
        "sdr": mean("sdr"),
        "sir": mean("sir"),
        "sar": mean("sar"),
        "rtf": mean("rtf"),
        "skip_ratio": mean("skip_ratio"),
        "n": len(data),
    })

out_csv = Path("results/summary_mini_enrollment_robustness.csv")
out_md = Path("results/summary_mini_enrollment_robustness.md")
headers = ["variant", "si_sdri", "sdr", "sir", "sar", "rtf", "skip_ratio", "n"]
with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    vals = []
    for h in headers:
        v = row[h]
        vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
    lines.append("| " + " | ".join(vals) + " |")
out_md.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out_csv} and {out_md}")
PY

for name in mini_exp01_base_converged mini_exp04_dynamic_converged; do
  cfg=configs/base_usef_tfgridnet.yaml
  if [[ "$name" == "mini_exp04_dynamic_converged" ]]; then
    cfg=configs/cafe_tse_dynamic.yaml
  fi
  "$PY" -m cafe_tse.cli.benchmark_efficiency \
    --config "$cfg" \
    --checkpoint "experiments/${name}/checkpoints/best.pt" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_csv "results/${name}_efficiency.csv" \
    --out_json "results/${name}_efficiency.json" \
    --device cuda \
    --num_samples 80 \
    --warmup 10 \
    "${EVAL_OVERRIDES[@]}"
done

"$PY" - <<'PY'
from pathlib import Path
import csv
import json

def md_table(csv_path):
    rows = list(csv.DictReader(Path(csv_path).open(newline="", encoding="utf-8")))
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
    return "\n".join(lines)

eff_rows = []
for name in ["mini_exp01_base_converged", "mini_exp04_dynamic_converged"]:
    data = json.loads(Path(f"results/{name}_efficiency.json").read_text(encoding="utf-8"))
    eff_rows.append({
        "method": name,
        "params": data["params"],
        "rtf_wall": f"{data['rtf_wall']:.6f}",
        "active_blocks": f"{data['active_blocks']:.3f}",
        "skip_ratio": f"{data['skip_ratio']:.6f}",
        "peak_memory_mb": f"{data['peak_memory_mb']:.2f}",
        "macs_thop": "" if data["macs_thop"] is None else f"{data['macs_thop']:.0f}",
        "active_macs_proxy": "" if data["active_macs_proxy"] is None else f"{data['active_macs_proxy']:.0f}",
    })
headers = list(eff_rows[0].keys())
with Path("results/summary_mini_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(eff_rows)
eff_md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in eff_rows:
    eff_md.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
Path("results/summary_mini_efficiency.md").write_text("\n".join(eff_md), encoding="utf-8")

report = [
    "# MiniLibriMix Real-Data Experiments",
    "",
    "Dataset: MiniLibriMix from the official Zenodo release (https://zenodo.org/records/3871592). Train uses 800 mixtures. The original val split is divided into non-overlapping validation and test subsets: 100 validation mixtures and 100 test mixtures.",
    "",
    "## Converged Training",
    "",
    md_table("results/summary_mini_converged_main.csv"),
    "",
    "## Enrollment Robustness",
    "",
    md_table("results/summary_mini_enrollment_robustness.csv"),
    "",
    "## Efficiency",
    "",
    "\n".join(eff_md),
    "",
    "Notes: `params` is the number of trainable parameters. `rtf_wall` is measured end-to-end inference time divided by audio duration on the RTX 4080 SUPER. Dynamic routing reduces active separator blocks and reports the corresponding skip ratio. If THOP cannot trace STFT/ISTFT, MAC fields are left blank and RTF/memory remain the primary hardware efficiency measurements.",
]
Path("results/report_analysis_mini.md").write_text("\n".join(report), encoding="utf-8")
print("wrote results/summary_mini_efficiency.md and results/report_analysis_mini.md")
PY

cat results/summary_mini_converged_main.md
printf '\n--- enrollment robustness ---\n'
cat results/summary_mini_enrollment_robustness.md
printf '\n--- efficiency ---\n'
cat results/summary_mini_efficiency.md
