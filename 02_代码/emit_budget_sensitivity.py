#!/usr/bin/env python3
"""Print the MVTec budget-sensitivity table and prose for the LaTeX manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value: float, plus: bool = False) -> str:
    return f"{value:+.4f}" if plus else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--format", choices=("tables", "prose", "json"), default="tables")
    args = parser.parse_args()
    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    summary = canonical["summary"]
    ordered = sorted(summary.items(), key=lambda kv: float(kv[0]))

    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.format == "prose":
        parts = []
        for budget, entry in ordered:
            parts.append(
                f"{float(budget)*100:.0f}\\% budget: AUROC difference "
                f"{_fmt(entry['auroc_delta_mean'], plus=True)} "
                f"(95\\% CI [{_fmt(entry['auroc_delta_ci'][0])}, "
                f"{_fmt(entry['auroc_delta_ci'][1])}]), "
                f"Recall difference {_fmt(entry['recall_delta_mean'], plus=True)} "
                f"([{_fmt(entry['recall_delta_ci'][0])}, "
                f"{_fmt(entry['recall_delta_ci'][1])}]), "
                f"with {entry['auroc_positive_count']}/45 AUROC-positive and "
                f"{entry['recall_positive_count']}/45 recall-positive units."
            )
        print(" ".join(parts))
        return

    print(
        "\\toprule "
        "Budget & $n$ & AUROC $\\Delta$ & 95\\% CI & $+$ units & "
        "Recall $\\Delta$ & 95\\% CI & Recall $+$ units \\\\"
    )
    for budget, entry in ordered:
        print(
            f"{float(budget)*100:.0f}\\% & {entry['unit_count']} & "
            f"{_fmt(entry['auroc_delta_mean'])} & "
            f"[{_fmt(entry['auroc_delta_ci'][0])}, {_fmt(entry['auroc_delta_ci'][1])}] & "
            f"{entry['auroc_positive_count']}/{entry['unit_count']} & "
            f"{_fmt(entry['recall_delta_mean'])} & "
            f"[{_fmt(entry['recall_delta_ci'][0])}, {_fmt(entry['recall_delta_ci'][1])}] & "
            f"{entry['recall_positive_count']}/{entry['unit_count']} \\\\"
        )
    print("\\bottomrule")


if __name__ == "__main__":
    main()
