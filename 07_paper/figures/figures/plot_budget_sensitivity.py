"""Publication figure: strict-budget local-patch routing sensitivity."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "Fig3_budget_sensitivity"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})


def main() -> None:
    df = pd.read_csv(HERE / "budget_sensitivity_source.csv")
    x = df["budget"].to_numpy() * 100
    colors = {"auroc": "#356A9A", "recall": "#C46A3A"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True)
    for panel, ax, metric, marker, linestyle in zip(
        ["(a)", "(b)"],
        axes,
        ["auroc", "recall"],
        ["o", "s"],
        ["-", "--"],
    ):
        y = df[f"{metric}_delta"].to_numpy()
        ax.plot(x, y, color=colors[metric], lw=1.8, marker=marker, linestyle=linestyle, ms=4.6,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
        low = df[f"{metric}_ci_low"].to_numpy()
        high = df[f"{metric}_ci_high"].to_numpy()
        mask = np.isfinite(low) & np.isfinite(high)
        if mask.any():
            ax.errorbar(x[mask], y[mask], yerr=[y[mask] - low[mask], high[mask] - y[mask]],
                        fmt="none", ecolor=colors[metric], elinewidth=1.2,
                        capsize=2.5, zorder=4)
        ax.axhline(0, color="#777777", lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(["10%", "25%", "50%"])
        ax.set_xlabel("Full-route fallback budget")
        ax.set_ylabel("Local Patch routing − matched random")
        ax.text(0.01, 0.98, panel, transform=ax.transAxes, va="top", fontweight="bold")
        ax.grid(axis="y", color="#D9DDE2", lw=0.5, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_xlim(5, 55)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
