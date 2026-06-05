#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/envs/cafe-tse/bin/python
export PYTHONPATH=src

META=data/metadata/minilibrimix_disjoint
BASE_CKPT=experiments/mini_exp01_base_converged/checkpoints/best.pt
STUDENT_CKPT=experiments/mini_exp10_distill_5block_mid/checkpoints/best.pt
OURS_NAME=mini_exp18_egsp_spec_s005_selected

EVAL_COMMON=(
  --override device=cuda
  --override sample_rate=8000
  --override segment_seconds=4.0
  --override batch_size=8
  --override num_workers=0
)

OURS_MODEL=(
  "${EVAL_COMMON[@]}"
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

BASE_EGSP_MODEL=(
  "${EVAL_COMMON[@]}"
  --override model.dynamic_inference=false
  --override model.egsp_enabled=true
  --override model.egsp_strength=0.05
  --override model.egsp_min_weight=0.80
  --override model.egsp_max_weight=1.20
  --override model.egsp_apply_to_spec=true
)

echo "[0/6] Smoke-check updated evaluation code"
"$PY" -m py_compile \
  src/cafe_tse/metrics/separation.py \
  src/cafe_tse/engine/evaluator.py \
  src/cafe_tse/cli/prepare_mixture_noise_variants.py

eval_base() {
  local name="$1"
  local save_audio="${2:-0}"
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/base_usef_tfgridnet.yaml \
    --checkpoint "$BASE_CKPT" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio "$save_audio" \
    --device cuda \
    "${EVAL_COMMON[@]}"
}

eval_student() {
  local name="$1"
  local save_audio="${2:-0}"
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$STUDENT_CKPT" \
    --test_manifest "$META/test_manifest_final.csv" \
    --out_dir "results/${name}" \
    --save_audio "$save_audio" \
    --device cuda \
    "${EVAL_COMMON[@]}" \
    --override model.emb_dim=40 \
    --override model.hidden_dim=160 \
    --override model.n_heads=4 \
    --override model.n_blocks=5 \
    --override model.sparse_fusion_blocks='[0,2,4]' \
    --override model.dynamic_inference=false \
    --override model.full_blocks=5
}

eval_ours_manifest() {
  local name="$1"
  local manifest="$2"
  local save_audio="${3:-0}"
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/cafe_tse_dynamic.yaml \
    --checkpoint "$STUDENT_CKPT" \
    --test_manifest "$manifest" \
    --out_dir "results/${name}" \
    --save_audio "$save_audio" \
    --device cuda \
    "${OURS_MODEL[@]}"
}

echo "[1/6] Re-evaluate main methods with PESQ/STOI"
eval_base mini_exp01_base_converged 8
eval_student mini_exp10_distill_5block_mid 8
eval_ours_manifest "$OURS_NAME" "$META/test_manifest_final.csv" 10

echo "[2/6] Probe 6-block baseline + EGSP quality variant"
"$PY" -m cafe_tse.cli.evaluate \
  --config configs/base_usef_tfgridnet.yaml \
  --checkpoint "$BASE_CKPT" \
  --test_manifest "$META/test_manifest_final.csv" \
  --out_dir results/mini_exp25_baseline_egsp_quality \
  --save_audio 8 \
  --device cuda \
  "${BASE_EGSP_MODEL[@]}"

"$PY" -m cafe_tse.cli.benchmark_efficiency \
  --config configs/base_usef_tfgridnet.yaml \
  --checkpoint "$BASE_CKPT" \
  --test_manifest "$META/test_manifest_final.csv" \
  --out_csv results/mini_exp25_baseline_egsp_quality_efficiency.csv \
  --out_json results/mini_exp25_baseline_egsp_quality_efficiency.json \
  --device cuda \
  --num_samples 100 \
  --warmup 5 \
  "${BASE_EGSP_MODEL[@]}"

echo "[3/6] Rebuild enrollment robustness for final Ours"
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
  eval_ours_manifest "guide_ours_${variant}" "$manifest" 0
done < "$META/enrollment_variants/variants.csv"

echo "[4/6] Build babble-noise mixture variants and evaluate robustness"
"$PY" -m cafe_tse.cli.prepare_mixture_noise_variants \
  --manifest "$META/test_manifest_final.csv" \
  --out_dir "$META/mixture_noise_variants" \
  --sample_rate 8000 \
  --noise_types babble \
  --snrs 5 10

while IFS=, read -r variant manifest noise_type snr; do
  if [[ "$variant" == "variant" ]]; then
    continue
  fi
  eval_ours_manifest "guide_ours_${variant}" "$manifest" 0
  "$PY" -m cafe_tse.cli.evaluate \
    --config configs/base_usef_tfgridnet.yaml \
    --checkpoint "$BASE_CKPT" \
    --test_manifest "$manifest" \
    --out_dir "results/guide_baseline_${variant}" \
    --save_audio 0 \
    --device cuda \
    "${EVAL_COMMON[@]}"
done < "$META/mixture_noise_variants/variants.csv"

echo "[5/6] Package demo audio comparisons"
"$PY" - <<'PY'
from pathlib import Path
import csv
import shutil

manifest = Path("data/metadata/minilibrimix_disjoint/test_manifest_final.csv")
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))[:3]
out = Path("demo_audio")
out.mkdir(exist_ok=True)
for old in out.glob("*.wav"):
    old.unlink()

def copy(src, dst):
    src = Path(src)
    shutil.copy2(src, out / dst)

for idx, row in enumerate(rows, 1):
    utt = row["utt_id"]
    prefix = f"case{idx:02d}_{utt}"
    copy(row["mixture_path"], f"{prefix}_mixture.wav")
    copy(row["target_path"], f"{prefix}_target.wav")
    copy(f"results/mini_exp01_base_converged/audio/{utt}_estimated.wav", f"{prefix}_baseline.wav")
    copy(f"results/mini_exp18_egsp_spec_s005_selected/audio/{utt}_estimated.wav", f"{prefix}_ours.wav")
print(f"wrote demo audio to {out.resolve()}")
PY

echo "[6/6] Summarize guide-oriented follow-up results"
"$PY" - <<'PY'
from pathlib import Path
import csv
import json

def load_summary(name):
    p = Path(f"results/{name}/summary.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    eff_path = Path(f"results/{name}_efficiency.json")
    eff = json.loads(eff_path.read_text(encoding="utf-8")) if eff_path.exists() else {}
    return {
        "method": name,
        "si_sdr": f"{data.get('si_sdr', float('nan')):.6f}",
        "si_sdri": f"{data.get('si_sdri', float('nan')):.6f}",
        "sdr": f"{data.get('sdr', float('nan')):.6f}",
        "sir": f"{data.get('sir', float('nan')):.6f}",
        "sar": f"{data.get('sar', float('nan')):.6f}",
        "stoi": f"{data.get('stoi', float('nan')):.6f}",
        "pesq": f"{data.get('pesq', float('nan')):.6f}",
        "rtf_eval": f"{data.get('rtf', float('nan')):.6f}",
        "params": data.get("params", ""),
        "rtf_wall": f"{eff.get('rtf_wall', float('nan')):.6f}" if eff else "",
        "active_blocks": f"{eff.get('active_blocks', float('nan')):.3f}" if eff else "",
        "macs": f"{eff.get('macs_thop', float('nan')):.0f}" if eff and eff.get("macs_thop") is not None else "",
    }

main_methods = [
    "mini_exp01_base_converged",
    "mini_exp10_distill_5block_mid",
    "mini_exp18_egsp_spec_s005_selected",
    "mini_exp25_baseline_egsp_quality",
]

enrollment_methods = [
    "guide_ours_enroll_1s_clean",
    "guide_ours_enroll_1s_noisy_10db",
    "guide_ours_enroll_3s_clean",
    "guide_ours_enroll_3s_noisy_10db",
    "guide_ours_enroll_5s_clean",
    "guide_ours_enroll_5s_noisy_10db",
]

noise_methods = [
    "guide_baseline_mixture_babble_5db",
    "guide_ours_mixture_babble_5db",
    "guide_baseline_mixture_babble_10db",
    "guide_ours_mixture_babble_10db",
]

groups = {
    "summary_mini_guide_main.csv": main_methods,
    "summary_mini_guide_enrollment.csv": enrollment_methods,
    "summary_mini_guide_mixture_noise.csv": noise_methods,
}

for filename, methods in groups.items():
    rows = [load_summary(name) for name in methods if Path(f"results/{name}/summary.json").exists()]
    if not rows:
        continue
    headers = list(rows[0].keys())
    csv_path = Path("results") / filename
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    md_path = csv_path.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n## {md_path}")
    print(md_path.read_text(encoding="utf-8"))
PY
