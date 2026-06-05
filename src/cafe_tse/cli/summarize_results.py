from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dirs", nargs="+", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()
    rows = []
    for result_dir in args.result_dirs:
        path = Path(result_dir) / "metrics.csv"
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        def mean(key: str) -> float:
            vals = [float(r[key]) for r in data if r.get(key) not in ("", "nan", None)]
            return sum(vals) / max(len(vals), 1)
        rows.append(
            {
                "Method": Path(result_dir).name,
                "SI-SDRi up": mean("si_sdri"),
                "SDR up": mean("sdr"),
                "SIR up": mean("sir"),
                "SAR up": mean("sar"),
                "Params down": data[0].get("params", "") if data else "",
                "RTF down": mean("rtf"),
                "Skip Ratio up": mean("skip_ratio") if data and "skip_ratio" in data[0] else "",
            }
        )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else ["Method"]
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_csv} and {args.out_md}")


if __name__ == "__main__":
    main()
