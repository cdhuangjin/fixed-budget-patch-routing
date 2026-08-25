from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

blue = "#477AA8"
orange = "#C96B42"
grey = "#AAB8C7"
dark = "#3C4650"

datasets = ["MVTec AD", "MPDD", "VisA"]
auroc = np.array([0.0783, 0.1013, 0.0124])
auroc_low = np.array([0.0525, 0.0593, 0.0079])
auroc_high = np.array([0.1029, 0.1448, 0.0166])
recall = np.array([0.1848, 0.2074, 0.2283])
recall_low = np.array([0.1192, 0.1208, 0.1483])
recall_high = np.array([0.2491, 0.3001, 0.3056])

budgets = np.array([10, 25, 50])
budget_auroc = np.array([0.0511, 0.0783, 0.1122])
budget_recall = np.array([0.2853, 0.1848, 0.1615])

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
y = np.arange(len(datasets))
for a, vals, low, high, label in [
    (ax[0], auroc, auroc_low, auroc_high, "AUROC difference"),
    (ax[1], recall, recall_low, recall_high, "Recall@5%FPR difference"),
]:
    a.axvline(0, color=dark, lw=0.8, ls="--")
    a.errorbar(vals, y, xerr=np.vstack([vals-low, high-vals]), fmt="o", ms=5,
               color=orange, ecolor=orange, elinewidth=1.8, capsize=3)
    a.set_yticks(y, datasets)
    a.set_xlabel(label)
    a.set_xlim(left=min(0, float(low.min()) - 0.01), right=float(high.max()) + 0.02)
    a.grid(axis="x", color="#D9DEE5", lw=0.6)
    a.set_axisbelow(True)
    a.tick_params(axis="both", length=3)
ax[0].set_title("Quality gain across datasets", loc="left", fontweight="bold")
ax[1].set_title("Low-FPR gain across datasets", loc="left", fontweight="bold")
fig.suptitle("Local-patch routing improves fixed-budget allocation", fontweight="bold", y=1.03)
save(fig, "Fig1_cross_dataset_effects")

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
ax[0].plot(budgets, budget_auroc, marker="o", lw=2.2, color=blue)
ax[0].axhline(0, color=dark, lw=0.8, ls="--")
ax[0].set_title("Budget sensitivity", loc="left", fontweight="bold")
ax[0].set_xlabel("Full-route fallback budget (%)")
ax[0].set_ylabel("AUROC difference vs matched random")
ax[0].set_xticks(budgets)
ax[0].grid(axis="y", color="#D9DEE5", lw=0.6)
ax[0].set_axisbelow(True)

labels = ["Local probe", "Full", "Sequential\nmixed"]
p95 = [3.127, 4.714, 7.135]
bars = ax[1].bar(labels, p95, color=[grey, blue, orange], width=0.62)
ax[1].set_title("Batch-one latency boundary", loc="left", fontweight="bold")
ax[1].set_ylabel("P95 latency (ms)")
ax[1].grid(axis="y", color="#D9DEE5", lw=0.6)
ax[1].set_axisbelow(True)
for b, val in zip(bars, p95):
    ax[1].text(b.get_x()+b.get_width()/2, val+0.35, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
fig.suptitle("Quality gains persist across budgets, but not as single-sample acceleration", fontweight="bold", y=1.03)
save(fig, "Fig2_budget_latency_boundary")

fig, ax = plt.subplots(figsize=(5.2, 2.8), constrained_layout=True)
categories = ["MVTec\nonline", "MPDD\nonline", "VisA\nonline", "VisA\nbuffer=128"]
online_auroc = [0.0331, 0.0890, 0.00004, 0.00124]
online_recall = [0.0804, 0.0643, -0.0408, 0.0750]
x = np.arange(len(categories))
w = 0.36
ax.bar(x-w/2, online_auroc, w, label="AUROC", color=blue)
ax.bar(x+w/2, online_recall, w, label="Recall@5%FPR", color=orange)
ax.axhline(0, color=dark, lw=0.8)
ax.set_xticks(x, categories)
ax.set_ylabel("Difference vs matched random")
ax.set_title("Online routing is configuration-sensitive", loc="left", fontweight="bold")
ax.set_ylim(-0.05, 0.11)
ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.72), borderaxespad=0)
ax.grid(axis="y", color="#D9DEE5", lw=0.6)
ax.set_axisbelow(True)
save(fig, "Fig3_online_sensitivity")

print("Generated", OUT)
