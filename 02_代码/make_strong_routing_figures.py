#!/usr/bin/env python3
"""Render Canonical V3 strong-routing main and delta figures."""

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

METHODS = ("random_auroc_mean", "fast_score_auroc_mean", "uncertainty_auroc_mean", "risk_auroc_mean")
LABELS = ("Random", "Fast-score", "Uncertainty", "Local-patch Risk")
COLORS = ("#4F5D75", "#D98F39", "#7C9EB2", "#A13D6F")
HATCHES = ("", "///", "xx", "..")
COMPARISONS = ("risk_random", "risk_fast_score", "risk_uncertainty")
COMPARISON_LABELS = ("Risk - Random", "Risk - Fast-score", "Risk - Uncertainty")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty table: {path}")
    return rows


def _save_publication_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figures(canonical_dir: Path, output_dir: Path) -> dict[str, Path]:
    rows = _read_csv(canonical_dir / "main_results.csv")
    expected = {"MVTec", "MPDD", "VisA"}
    if {row["dataset"] for row in rows} != expected:
        raise ValueError("main_results.csv must contain MVTec, MPDD and VisA")
    if any(int(row["unit_count"]) <= 0 for row in rows):
        raise ValueError("main_results.csv unit_count must be positive")

    ordered = {row["dataset"]: row for row in rows}
    datasets = ("MVTec", "MPDD", "VisA")
    x = np.arange(len(datasets))
    width = 0.19

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    ax = axes[0]
    for offset, method, label, color, hatch in zip(
        (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width,
        METHODS, LABELS, COLORS, HATCHES
    ):
        values = [float(ordered[dataset][method]) for dataset in datasets]
        ax.bar(
            x + offset, values, width=width, label=label, color=color,
            hatch=hatch, edgecolor="#333333", linewidth=0.45, zorder=3,
        )
    ax.set_xticks(x, [f"{dataset}\nn={ordered[dataset]['unit_count']}" for dataset in datasets])
    ax.set_ylabel("Image AUROC")
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, va="top", fontweight="bold")
    ax.set_ylim(0, 1.04)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        borderaxespad=0,
    )

    ax = axes[1]
    markers = ("o", "s", "^")
    for index, (comparison, label, marker) in enumerate(zip(COMPARISONS, COMPARISON_LABELS, markers)):
        means = np.array([float(ordered[dataset][f"{comparison}_delta_mean"]) for dataset in datasets])
        lows = np.array([float(ordered[dataset][f"{comparison}_delta_low"]) for dataset in datasets])
        highs = np.array([float(ordered[dataset][f"{comparison}_delta_high"]) for dataset in datasets])
        offset = (index - 1) * 0.16
        ax.errorbar(
            x + offset, means, yerr=np.vstack((means - lows, highs - means)),
            fmt=marker, markersize=4, capsize=2.5, linewidth=1.0, label=label,
        )
    ax.axhline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Paired AUROC difference")
    ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, va="top", fontweight="bold")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.6,
        borderaxespad=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_canonical_v3_strong_routing"
    _save_publication_figure(fig, output_path)
    return {"strong_routing": output_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = generate_figures(args.canonical_dir, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
