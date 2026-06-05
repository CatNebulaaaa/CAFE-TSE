from __future__ import annotations

import csv
import json
from pathlib import Path


def md_table(csv_path: str) -> str:
    rows = list(csv.DictReader(Path(csv_path).open(newline="", encoding="utf-8")))
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "") for h in headers) + " |")
    return "\n".join(lines)


def write_efficiency_summary() -> None:
    rows = []
    for name in ["mini_exp01_base_converged", "mini_exp04_dynamic_converged"]:
        data = json.loads(Path(f"results/{name}_efficiency.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "method": name,
                "params": data["params"],
                "rtf_wall": f"{data['rtf_wall']:.6f}",
                "active_blocks": f"{data['active_blocks']:.3f}",
                "skip_ratio": f"{data['skip_ratio']:.6f}",
                "peak_memory_mb": f"{data['peak_memory_mb']:.2f}",
                "macs_thop": "" if data["macs_thop"] is None else f"{data['macs_thop']:.0f}",
                "active_macs_proxy": "" if data["active_macs_proxy"] is None else f"{data['active_macs_proxy']:.0f}",
            }
        )
    headers = list(rows[0].keys())
    with Path("results/summary_mini_efficiency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    Path("results/summary_mini_efficiency.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    write_efficiency_summary()
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
        md_table("results/summary_mini_efficiency.csv"),
        "",
        "Notes: `params` is the number of trainable parameters. `rtf_wall` is measured end-to-end inference time divided by audio duration on the RTX 4080 SUPER. SDR/SIR/SAR use a two-source projection metric implemented without SciPy to avoid platform `libstdc++` conflicts.",
    ]
    Path("results/report_analysis_mini.md").write_text("\n".join(report), encoding="utf-8")
    print("regenerated MiniLibriMix summaries and report")


if __name__ == "__main__":
    main()
