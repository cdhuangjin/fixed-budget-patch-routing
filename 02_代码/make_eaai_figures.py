from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "07_论文" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / "source_data_eaai_figures.csv"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
    }
)

COLORS = {
    "Fast": "#8C96A6",
    "Random": "#B9C4D0",
    "Risk": "#D47A63",
    "Full": "#4E7896",
}


def save_pub(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax, label):
    ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def main_figure(df):
    main = df[df["figure"] == "main"].copy()
    budget = df[df["figure"] == "budget"].copy()
    latency = df[df["figure"] == "latency"].copy()

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    methods = ["Fast", "Random", "Risk", "Full"]
    x = np.arange(len(methods))
    width = 0.36
    ax_a.bar(x - width / 2, [main.loc[main.method == m, "auroc"].iloc[0] for m in methods], width,
             color=[COLORS[m] for m in methods], edgecolor="white", linewidth=0.5, label="AUROC")
    ax_a.bar(x + width / 2, [main.loc[main.method == m, "recall"].iloc[0] for m in methods], width,
             color=[COLORS[m] for m in methods], edgecolor="black", linewidth=0.5, alpha=0.42,
             hatch="//", label="Recall@5%FPR")
    ax_a.set_xticks(x, methods)
    ax_a.set_ylim(0.35, 1.02)
    ax_a.set_ylabel("Macro mean")
    ax_a.set_title("Matched-budget comparison", loc="left", fontsize=8)
    ax_a.legend(ncol=2, fontsize=6, handlelength=1.2, columnspacing=0.8)
    add_panel_label(ax_a, "a")

    for metric, ylabel, ax in [("auroc", "AUROC", ax_b), ("recall", "Recall@5%FPR", ax_c)]:
        for method in ["Random", "Risk"]:
            sub = budget[budget.method == method]
            ax.plot(sub.target_budget, sub[metric], marker="o", markersize=3.8, linewidth=1.5,
                    color=COLORS[method], label=method)
        ax.set_xticks([10, 25, 40], ["10%", "25%", "40%"])
        ax.set_xlabel("Target calibration budget")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.45 if metric == "recall" else 0.68, 0.92)
        ax.grid(axis="y", color="#D9DEE5", linewidth=0.5, alpha=0.7)
        ax.legend(fontsize=6, loc="lower right")
    ax_b.set_title("Risk advantage persists across budgets", loc="left", fontsize=8)
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")

    labels = ["Fast", "Full", "Risk"]
    xx = np.arange(len(labels))
    ax_d.bar(xx - width / 2, [latency.loc[latency.method == m, "p50_ms"].iloc[0] for m in labels], width,
             color=[COLORS[m] for m in labels], label="P50")
    ax_d.bar(xx + width / 2, [latency.loc[latency.method == m, "p95_ms"].iloc[0] for m in labels], width,
             color=[COLORS[m] for m in labels], alpha=0.45, hatch="//", label="P95")
    ax_d.set_xticks(xx, labels)
    ax_d.set_ylabel("Latency (ms)")
    ax_d.set_title("Unified batch-one CUDA latency", loc="left", fontsize=8)
    ax_d.legend(fontsize=6)
    ax_d.grid(axis="y", color="#D9DEE5", linewidth=0.5, alpha=0.7)
    add_panel_label(ax_d, "d")

    fig.suptitle("Risk-aware escalation improves budgeted anomaly detection over matched random escalation",
                 x=0.02, ha="left", fontsize=9, fontweight="bold")
    save_pub(fig, "Fig1_main_results")


def uncertainty_figure(df):
    sub = df[df["figure"] == "uncertainty"].copy()
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    fig.subplots_adjust(bottom=0.30, left=0.28, right=0.98, top=0.82)
    y = np.arange(len(sub))
    ax.errorbar(sub.estimate, y, xerr=[sub.estimate - sub.ci_low, sub.ci_high - sub.estimate],
                fmt="o", color="#D47A63", ecolor="#D47A63", elinewidth=1.5, capsize=3, markersize=4)
    ax.axvline(0, color="#59636E", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, sub.method)
    ax.set_xlabel("Risk minus matched Random", labelpad=9)
    ax.set_title("Paired bootstrap uncertainty", loc="left", fontsize=8)
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.5, alpha=0.7)
    add_panel_label(ax, "a")
    fig.text(0.28, 0.035, "n = 45 category–seed units; 95% CI from paired bootstrap resampling.", fontsize=6)
    save_pub(fig, "Fig2_risk_random_uncertainty")


def write_source_data():
    rows = [
        ["main", "Fast", "", 0.7860, 0.4575, "", ""],
        ["main", "Random", "", 0.7796, 0.6204, "", ""],
        ["main", "Risk", "", 0.8674, 0.7251, "", ""],
        ["main", "Full", "", 0.9390, 0.7852, "", ""],
        ["budget", "Random", 10, 0.7379, 0.5222, "", ""],
        ["budget", "Risk", 10, 0.8322, 0.6404, "", ""],
        ["budget", "Random", 25, 0.7889, 0.6396, "", ""],
        ["budget", "Risk", 25, 0.8652, 0.7269, "", ""],
        ["budget", "Random", 40, 0.8264, 0.7035, "", ""],
        ["budget", "Risk", 40, 0.8886, 0.7512, "", ""],
        ["latency", "Fast", "", "", "", 2.44, 7.35],
        ["latency", "Full", "", "", "", 4.09, 10.52],
        ["latency", "Risk", "", "", "", 5.85, 16.47],
        ["uncertainty", "AUROC delta", "", "", "", 0.0878722866, ""],
        ["uncertainty", "Recall delta", "", "", "", 0.1047240539, ""],
    ]
    frame = pd.DataFrame(rows, columns=["figure", "method", "target_budget", "auroc", "recall", "p50_ms", "p95_ms"])
    frame["estimate"] = np.nan
    frame["ci_low"] = np.nan
    frame["ci_high"] = np.nan
    frame.loc[frame.method == "AUROC delta", ["estimate", "ci_low", "ci_high"]] = [0.0878722866, 0.0676277243, 0.1081036943]
    frame.loc[frame.method == "Recall delta", ["estimate", "ci_low", "ci_high"]] = [0.1047240539, 0.0754212729, 0.1337446796]
    frame.to_csv(DATA, index=False)
    return frame


if __name__ == "__main__":
    source = write_source_data()
    main_figure(source)
    uncertainty_figure(source)
    print(f"Wrote figures and source data to {OUT}")
