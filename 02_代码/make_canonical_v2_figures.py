#!/usr/bin/env python3
"""Render Canonical V2 allocation and separate latency-audit figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

SIGNAL = "#A13D6F"
NEUTRAL = "#4F5D75"
ACCENT = "#D98F39"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save_publication_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_allocation_effect(stats_rows: list[dict[str, str]], output_path: Path) -> None:
    datasets = [f"{row['dataset']}\nn={row['unit_count']}" for row in stats_rows]
    means = np.array([float(row["mean_delta"]) for row in stats_rows])
    lows = np.array([float(row["ci95_low"]) for row in stats_rows])
    highs = np.array([float(row["ci95_high"]) for row in stats_rows])
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.errorbar(
        x, means, yerr=np.vstack((means - lows, highs - means)), fmt="o",
        color=SIGNAL, markersize=5, capsize=3, linewidth=1.1, zorder=3,
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("AUROC difference\n(Risk - matched Random)")
    lower = min(-0.01, float(lows.min()) - 0.01)
    upper = float(highs.max()) + 0.03
    ax.set_ylim(lower, upper)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    fig.tight_layout()
    _save_publication_figure(fig, output_path)


def _plot_latency_audit(efficiency_rows: list[dict[str, str]], output_path: Path) -> None:
    display_order = ("fast", "full", "risk")
    indexed = {row["path"]: row for row in efficiency_rows}
    labels = ["Fast", "Full", "Risk route"]
    means = [float(indexed[path]["mean_ms"]) for path in display_order]
    p95s = [float(indexed[path]["p95_ms"]) for path in display_order]
    x = np.arange(len(display_order))
    width = 0.33

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    mean_bars = ax.bar(
        x - width / 2, means, width, label="Mean", color=NEUTRAL,
        hatch="", edgecolor="#333333", linewidth=0.45, zorder=3,
    )
    p95_bars = ax.bar(
        x + width / 2, p95s, width, label="P95", color=ACCENT,
        hatch="///", edgecolor="#333333", linewidth=0.45, zorder=3,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("End-to-end latency (ms)")
    fig.legend([mean_bars, p95_bars], ["Mean", "P95"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.set_xlabel("Path")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    _save_publication_figure(fig, output_path)


def generate_figures(canonical_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Generate figures solely from the materialized Canonical V2 CSV tables."""
    stats_rows = _read_csv(canonical_dir / "stats_results.csv")
    if not stats_rows:
        raise ValueError("canonical stats table is empty")
    efficiency_rows = _read_csv(canonical_dir / "efficiency_results.csv")
    if not efficiency_rows or "path" not in efficiency_rows[0]:
        raise ValueError("canonical directory has no audited efficiency table")
    required_paths = {"fast", "full", "risk"}
    if {row.get("path") for row in efficiency_rows} != required_paths:
        raise ValueError("audited efficiency table must contain fast, full, and risk paths")
    if any(row.get("comparison_scope") != "separate_system_audit_not_paired_with_accuracy_rows" for row in efficiency_rows):
        raise ValueError("audited efficiency rows must remain separate from accuracy results")

    output_dir.mkdir(parents=True, exist_ok=True)
    allocation_output = output_dir / "fig_canonical_v2_allocation_effect"
    latency_output = output_dir / "fig_canonical_v2_latency_audit"
    _plot_allocation_effect(stats_rows, allocation_output)
    _plot_latency_audit(efficiency_rows, latency_output)
    return {"allocation_effect": allocation_output, "latency_audit": latency_output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    outputs = generate_figures(args.canonical_dir, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
