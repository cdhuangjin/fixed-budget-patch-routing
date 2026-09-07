#!/usr/bin/env python3
"""Append Fig3 budget-sensitivity rows to the submission figure source-data CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    summary = canonical["summary"]
    ordered = sorted(summary.items(), key=lambda kv: float(kv[0]))

    existing = []
    fields = ["figure", "panel", "dataset_or_condition", "metric", "value",
              "ci_low", "ci_high", "n", "notes"]
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    existing = [row for row in existing if row.get("figure") != "Fig3_budget_sensitivity"]

    new_rows = []
    for budget, entry in ordered:
        label = f"{float(budget)*100:.0f}% budget"
        new_rows.append({
            "figure": "Fig3_budget_sensitivity",
            "panel": "AUROC delta",
            "dataset_or_condition": "MVTec AD",
            "metric": f"{label}: Risk minus Random",
            "value": _fmt(entry["auroc_delta_mean"]),
            "ci_low": _fmt(entry["auroc_delta_ci"][0]),
            "ci_high": _fmt(entry["auroc_delta_ci"][1]),
            "n": entry["unit_count"],
            "notes": "paired bootstrap, 3 seeds",
        })
        new_rows.append({
            "figure": "Fig3_budget_sensitivity",
            "panel": "Recall delta",
            "dataset_or_condition": "MVTec AD",
            "metric": f"{label}: Risk minus Random",
            "value": _fmt(entry["recall_delta_mean"]),
            "ci_low": _fmt(entry["recall_delta_ci"][0]),
            "ci_high": _fmt(entry["recall_delta_ci"][1]),
            "n": entry["unit_count"],
            "notes": "paired bootstrap, 3 seeds",
        })

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(new_rows)
    print(f"wrote {len(new_rows)} Fig3 rows to {args.output}")


if __name__ == "__main__":
    main()
