#!/usr/bin/env python3
"""Print the LaTeX main-table rows from the canonical V3 summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _fmt(value: str, plus: bool = False) -> str:
    number = float(value)
    return f"{number:+.4f}" if plus else f"{number:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--format", choices=("tables", "json"), default="tables")
    args = parser.parse_args()

    summary_path = args.canonical_dir / "main_results.csv"
    raw_path = args.canonical_dir / "raw_results.csv"
    with summary_path.open(encoding="utf-8") as handle:
        summary_rows = {row["dataset"]: row for row in csv.DictReader(handle)}

    with raw_path.open(encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))

    def order(dataset_upper: str) -> int:
        return {"MVTEC": 0, "MPDD": 1, "VISA": 2}[dataset_upper]

    datasets_sorted = sorted(summary_rows, key=lambda d: order(d.upper()))
    labels = {"MVTec": "MVTec AD", "MPDD": "MPDD", "VisA": "VisA"}
    recall_fields = ["fast_only_recall", "risk_recall", "random_recall",
                     "fast_score_recall", "uncertainty_recall"]

    if args.format == "json":
        for dataset in datasets_sorted:
            rows = [row for row in raw_rows if row["dataset"] == dataset]
            recall_mean = {}
            for field in recall_fields:
                values = [float(row[field]) for row in rows]
                recall_mean[field] = sum(values) / len(values)
            print(json.dumps({"dataset": dataset, "rows": summary_rows[dataset], "recall_mean": recall_mean}))
        return

    for dataset in datasets_sorted:
        row = summary_rows[dataset]
        raw = [r for r in raw_rows if r["dataset"] == dataset]
        recall = {field: sum(float(r[field]) for r in raw) / len(raw) for field in recall_fields}
        print(
            f"{labels[dataset]} & {row['unit_count']} & "
            f"{_fmt(row['fast_only_auroc_mean'])} & {_fmt(row['risk_auroc_mean'])} & "
            f"{_fmt(recall['risk_recall'])} & {_fmt(row['random_auroc_mean'])} & "
            f"{_fmt(recall['random_recall'])} & {_fmt(row['fast_score_auroc_mean'])} & "
            f"{_fmt(recall['fast_score_recall'])} & {_fmt(row['uncertainty_auroc_mean'])} & "
            f"{_fmt(recall['uncertainty_recall'])} & "
            f"{_fmt(row['risk_random_delta_mean'], plus=True)} & "
            f"[{_fmt(row['risk_random_delta_low'])}, {_fmt(row['risk_random_delta_high'])}] & "
            f"{_fmt(row['risk_fast_score_delta_mean'], plus=True)} & "
            f"[{_fmt(row['risk_fast_score_delta_low'])}, {_fmt(row['risk_fast_score_delta_high'])}] \\\\"
        )


if __name__ == "__main__":
    main()
