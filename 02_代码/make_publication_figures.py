"""Generate publication-ready figures for 027 experiment."""
import json, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录")
OUT = ROOT / "一区候选_027_phase1_multiseed"
OUT.mkdir(parents=True, exist_ok=True)

# Load MVTec data
mvtec_dir = ROOT / "027_full_15class"
mvtec_cats = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
              "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]

records = []
for cat in mvtec_cats:
    f = mvtec_dir / f"{cat}.json"
    if f.exists():
        r = json.loads(f.read_text(encoding="utf-8"))
        records.append(r)

# === Figure 1: Efficiency Pareto (latency vs accuracy) ===
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
methods = [
    ("fast_only", "Fast Only", "o", "#2196F3"),
    ("strict_quota", "Strict Quota (25%)", "s", "#FF9800"),
    ("strict_quota_random", "Strict Quota Random", "s", "#FF9800", 0.5),
    ("random_fallback", "Random Fallback", "^", "#4CAF50"),
    ("risk_fallback", "Risk Fallback (Ours)", "D", "#E91E63"),
]
for item in methods:
    method_key, method_name, marker, color = item[0], item[1], item[2], item[3]
    alpha = item[4] if len(item) > 4 else 1.0
    aurocs = [r[method_key]["image_auroc"] for r in records if method_key in r]
    latencies = [r[method_key]["mean_ms_estimated"] for r in records if method_key in r]
    if aurocs:
        ax.scatter(statistics.mean(latencies), statistics.mean(aurocs), 
                   s=150, marker=marker, c=color, alpha=alpha, label=method_name, zorder=5)
ax.set_xlabel("Mean Latency (ms)", fontsize=12)
ax.set_ylabel("AUROC", fontsize=12)
ax.set_title("027: Efficiency-Accuracy Pareto (MVTec AD)", fontsize=14)
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(str(OUT / "fig_pareto_mvtec.png"), dpi=150)
plt.close(fig)
print("Saved fig_pareto_mvtec.png")

# === Figure 2: Hard-set vs Easy-set ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

hard_cats = [r["category"] for r in records if r["fast_only"]["image_auroc"] < 0.90]
easy_cats = [r["category"] for r in records if r["fast_only"]["image_auroc"] >= 0.90]

for idx, (cat_set, title) in enumerate([(hard_cats, "Hard Categories"), (easy_cats, "Easy Categories")]):
    ax = axes[idx]
    subset = [r for r in records if r["category"] in cat_set]
    x = np.arange(len(subset))
    width = 0.25
    
    fast = [r["fast_only"]["image_auroc"] for r in subset]
    risk = [r["risk_fallback"]["image_auroc"] for r in subset]
    rand = [r["random_fallback"]["image_auroc"] for r in subset]
    
    ax.bar(x - width, fast, width, label="Fast Only", color="#2196F3", alpha=0.8)
    ax.bar(x, risk, width, label="Risk Fallback", color="#E91E63", alpha=0.8)
    ax.bar(x + width, rand, width, label="Random Fallback", color="#4CAF50", alpha=0.8)
    
    ax.set_xlabel("Category", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([r["category"] for r in subset], rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(0.3, 1.05)

fig.suptitle("027: Per-Category Performance (MVTec AD)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(str(OUT / "fig_hard_easy_mvtec.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved fig_hard_easy_mvtec.png")

# === Figure 3: Risk Delta per category ===
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
cats = [r["category"] for r in records]
deltas = [r["risk_fallback"]["image_auroc"] - r["random_fallback"]["image_auroc"] for r in records]
colors = ["#E91E63" if d > 0.05 else "#FF9800" if d > 0 else "#2196F3" for d in deltas]
x = np.arange(len(cats))
ax.bar(x, deltas, color=colors, alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=9)
ax.set_ylabel("AUROC Delta (Risk - Random)", fontsize=12)
ax.set_title("027: Selective Inference Benefit per Category", fontsize=14)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=statistics.mean(deltas), color='red', linewidth=1, linestyle='--', label=f'Mean={statistics.mean(deltas):.3f}')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')
fig.tight_layout()
fig.savefig(str(OUT / "fig_delta_per_category.png"), dpi=150)
plt.close(fig)
print("Saved fig_delta_per_category.png")

print("\nAll figures saved to", OUT)
