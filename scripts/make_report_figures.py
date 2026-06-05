from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS = Path("results")
FIG_DIR = RESULTS / "figures"

METHOD_LABELS = {
    "mini_exp01_base_converged": "Baseline",
    "mini_exp10_distill_5block_mid": "5-block Student",
    "mini_exp18_egsp_spec_s005_selected": "Ours",
}

COLORS = {
    "Baseline": "#3b5b92",
    "5-block Student": "#2f8f83",
    "Ours": "#c15a36",
    "Ours-Fast": "#8a5fbf",
    "Gated+MR": "#d99b2b",
    "Dynamic Fusion": "#4f8fb8",
    "Full System": "#6a7f3f",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def write_md_table(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax) -> None:
    ax.grid(axis="y", color="#d8dde6", linewidth=0.8, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_main_metrics(main_rows: list[dict[str, str]]) -> None:
    metrics = ["si_sdri", "sdr", "sir", "sar"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        labels = [METHOD_LABELS[row["method"]] for row in main_rows]
        vals = [as_float(row[metric]) for row in main_rows]
        ax.bar(labels, vals, color=[COLORS[label] for label in labels], width=0.64)
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=35)
        style_axes(ax)
    fig.suptitle("Separation Metrics on MiniLibriMix Test", fontsize=13)
    fig.savefig(FIG_DIR / "main_metrics_bar.png", dpi=220)
    plt.close(fig)


def plot_efficiency(main_rows: list[dict[str, str]], eff_rows: list[dict[str, str]]) -> None:
    main_by = {row["method"]: row for row in main_rows}
    eff_by = {row["method"]: row for row in eff_rows}
    methods = [row["method"] for row in main_rows if row["method"] in eff_by]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    panels = [
        ("params", "Params"),
        ("macs_thop", "MACs"),
        ("rtf_wall", "RTF wall"),
    ]
    for ax, (key, title) in zip(axes, panels):
        labels = [METHOD_LABELS[m] for m in methods]
        if key == "params":
            vals = [as_float(main_by[m][key]) / 1000 for m in methods]
            ylabel = "K params"
        elif key == "macs_thop":
            vals = [as_float(eff_by[m][key]) / 1e9 for m in methods]
            ylabel = "G MACs"
        else:
            vals = [as_float(eff_by[m][key]) for m in methods]
            ylabel = "RTF"
        ax.bar(labels, vals, color=[COLORS[label] for label in labels], width=0.64)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        style_axes(ax)
    fig.suptitle("Efficiency Comparison", fontsize=13)
    fig.savefig(FIG_DIR / "efficiency_bar.png", dpi=220)
    plt.close(fig)


def plot_pareto(main_rows: list[dict[str, str]], eff_rows: list[dict[str, str]]) -> None:
    eff_by = {row["method"]: row for row in eff_rows}
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for row in main_rows:
        method = row["method"]
        label = METHOD_LABELS[method]
        macs = as_float(eff_by[method]["macs_thop"]) / 1e9
        si_sdri = as_float(row["si_sdri"])
        ax.scatter(macs, si_sdri, s=95, color=COLORS[label], label=label)
        ax.annotate(label, (macs, si_sdri), xytext=(6, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("MACs (G, lower is better)")
    ax.set_ylabel("SI-SDRi (higher is better)")
    ax.set_title("Accuracy-Efficiency Pareto View")
    style_axes(ax)
    fig.savefig(FIG_DIR / "pareto_macs_sisdr.png", dpi=220)
    plt.close(fig)


def plot_training_curve() -> None:
    rows = read_csv(RESULTS / "mini_exp10_distill_5block_mid_train_log.csv")
    epochs = [int(row["epoch"]) for row in rows]
    valid = [as_float(row["valid_loss"]) for row in rows]
    best_idx = min(range(len(valid)), key=lambda i: valid[i])
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.plot(epochs, valid, color="#2f8f83", linewidth=1.7)
    ax.scatter([epochs[best_idx]], [valid[best_idx]], color="#c15a36", s=55, zorder=3)
    ax.axvline(60, color="#727b8c", linestyle="--", linewidth=1.0)
    ax.text(60, max(valid), "60 epoch budget", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title("5-block Distillation Training Curve")
    style_axes(ax)
    fig.savefig(FIG_DIR / "training_curve_5block.png", dpi=220)
    plt.close(fig)


def difficulty_summary() -> list[dict[str, object]]:
    files = {
        "mini_exp01_base_converged": RESULTS / "mini_exp01_base_converged_metrics.csv",
        "mini_exp10_distill_5block_mid": RESULTS / "mini_exp10_distill_5block_mid_metrics.csv",
        "mini_exp18_egsp_spec_s005_selected": RESULTS / "mini_exp18_egsp_spec_s005_selected_metrics.csv",
    }
    rows = []
    for method, path in files.items():
        buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(path):
            buckets[row.get("difficulty", "unknown")].append(row)
        for difficulty in ["easy", "medium", "hard", "unknown"]:
            items = buckets.get(difficulty, [])
            if not items:
                continue
            out = {"method": METHOD_LABELS[method], "difficulty": difficulty, "n": len(items)}
            for metric in ["si_sdri", "sdr", "sir", "sar"]:
                vals = [as_float(row[metric]) for row in items]
                vals = [v for v in vals if not math.isnan(v)]
                out[metric] = f"{sum(vals) / max(len(vals), 1):.6f}"
            rows.append(out)
    save_csv(RESULTS / "summary_mini_difficulty.csv", rows)
    write_md_table(RESULTS / "summary_mini_difficulty.md", rows)
    return rows


def plot_difficulty(rows: list[dict[str, object]]) -> None:
    difficulties = [d for d in ["easy", "medium", "hard"] if any(row["difficulty"] == d for row in rows)]
    methods = ["Baseline", "5-block Student", "Ours"]
    width = 0.24
    x = list(range(len(difficulties)))
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for idx, method in enumerate(methods):
        vals = []
        for difficulty in difficulties:
            match = next((row for row in rows if row["method"] == method and row["difficulty"] == difficulty), None)
            vals.append(as_float(str(match["si_sdri"])) if match else 0.0)
        offset = (idx - 1) * width
        ax.bar([v + offset for v in x], vals, width=width, label=method, color=COLORS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(difficulties)
    ax.set_ylabel("SI-SDRi")
    ax.set_title("Difficulty-Aware SI-SDRi")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.savefig(FIG_DIR / "difficulty_sisdr_bar.png", dpi=220)
    plt.close(fig)


def plot_complexity_scatter() -> None:
    rows = read_csv(RESULTS / "mini_exp18_egsp_spec_s005_selected_metrics.csv")
    colors = {"easy": "#5aa469", "medium": "#d99b2b", "hard": "#c15a36"}
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for difficulty, color in colors.items():
        items = [row for row in rows if row.get("difficulty") == difficulty]
        ax.scatter(
            [as_float(row["complexity_score"]) for row in items],
            [as_float(row["si_sdri"]) for row in items],
            s=30,
            alpha=0.78,
            color=color,
            label=difficulty,
        )
    ax.set_xlabel("Complexity score")
    ax.set_ylabel("SI-SDRi")
    ax.set_title("Ours Performance by Complexity")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.savefig(FIG_DIR / "complexity_vs_sisdr_egsp.png", dpi=220)
    plt.close(fig)


def plot_system_innovations() -> None:
    path = RESULTS / "summary_mini_system_innovations.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    keep = [
        ("mini_exp01_base_converged", "Baseline"),
        ("mini_exp10_distill_5block_mid", "5-block Student"),
        ("mini_exp18_egsp_spec_s005_selected", "Ours"),
        ("mini_exp21_dynamic_sparse_fusion_eval", "Dynamic Fusion"),
        ("mini_exp22_gated_mrstft_finetune", "Gated+MR"),
        ("mini_exp23_depthaware_dynamic", "Ours-Fast"),
        ("mini_exp24_ours_system_full_static", "Full System"),
    ]
    by_method = {row["method"]: row for row in rows}
    selected = [by_method[method] for method, _ in keep if method in by_method]
    labels = [label for method, label in keep if method in by_method]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7), constrained_layout=True)
    panels = [
        ("si_sdri", "SI-SDRi", 1.0),
        ("rtf_wall", "RTF wall", 1.0),
        ("macs", "MACs (G)", 1e9),
    ]
    for ax, (key, title, scale) in zip(axes, panels):
        vals = [as_float(row[key]) / scale for row in selected]
        ax.bar(labels, vals, color=[COLORS.get(label, "#777777") for label in labels], width=0.62)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=32)
        style_axes(ax)
    fig.suptitle("System-Level Innovation Ablation on MiniLibriMix", fontsize=13)
    fig.savefig(FIG_DIR / "system_innovation_ablation.png", dpi=220)
    plt.close(fig)


def plot_egsp_sanity() -> None:
    path = RESULTS / "summary_mini_egsp_sanity.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    label_map = {
        "mini_exp10_distill_5block_mid": "5-block",
        "mini_exp18_egsp_spec_s005_selected": "EGSP correct",
        "mini_exp19_egsp_shuffled_enroll": "EGSP shuffled",
        "mini_exp20_egsp_enroll_1s_clean": "EGSP 1s enroll",
    }
    labels = [label_map.get(row["method"], row["method"]) for row in rows]
    sisdri = [as_float(row["si_sdri"]) for row in rows]
    sar = [as_float(row["sar"]) for row in rows]
    x = list(range(len(rows)))
    fig, ax1 = plt.subplots(figsize=(7.6, 4.0), constrained_layout=True)
    ax1.bar(x, sisdri, color="#c15a36", width=0.58, label="SI-SDRi")
    ax1.set_ylabel("SI-SDRi")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, sar, color="#3b5b92", marker="o", linewidth=1.8, label="SAR")
    ax2.set_ylabel("SAR")
    ax1.set_title("EGSP Sanity Check")
    style_axes(ax1)
    ax2.spines["top"].set_visible(False)
    fig.savefig(FIG_DIR / "egsp_sanity_check.png", dpi=220)
    plt.close(fig)


def plot_enrollment_robustness() -> None:
    path = RESULTS / "summary_mini_enrollment_robustness.csv"
    if not path.exists():
        return
    rows = read_csv(path)
    labels = [row.get("Method", row.get("method", "")) for row in rows]
    metric_key = "SI-SDRi up" if "SI-SDRi up" in rows[0] else "si_sdri"
    vals = [as_float(row[metric_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    ax.bar(labels, vals, color="#2f8f83", width=0.62)
    ax.set_ylabel("SI-SDRi")
    ax.set_title("Enrollment Robustness")
    ax.tick_params(axis="x", rotation=30)
    style_axes(ax)
    fig.savefig(FIG_DIR / "enrollment_robustness.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = read_csv(RESULTS / "summary_mini_egsp_selected_main.csv")
    eff_rows = read_csv(RESULTS / "summary_mini_egsp_selected_efficiency.csv")
    plot_main_metrics(main_rows)
    plot_efficiency(main_rows, eff_rows)
    plot_pareto(main_rows, eff_rows)
    plot_training_curve()
    diff_rows = difficulty_summary()
    plot_difficulty(diff_rows)
    plot_complexity_scatter()
    plot_system_innovations()
    plot_egsp_sanity()
    plot_enrollment_robustness()
    print(f"wrote figures to {FIG_DIR}")
    print((RESULTS / "summary_mini_difficulty.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
