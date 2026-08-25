"""027 Bootstrap confidence interval analysis."""
import json
import numpy as np
from pathlib import Path

def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    alpha = (1 - ci) / 2
    return float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))

def load_mvtec_results(base_dir):
    """Load MVTec per-category results across3 seeds."""
    categories = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", 
                  "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                  "transistor", "wood", "zipper"]
    
    results = {}
    for cat in categories:
        for seed in [5, 17, 29]:
            fpath = Path(base_dir) / f"mvtec-{cat}-seed{seed}" / "result.json"
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                key = f"{cat}_seed{seed}"
                results[key] = {
                    "category": cat,
                    "seed": seed,
                    "fast_auroc": data.get("fast_only", {}).get("image_auroc", 0),
                    "risk_auroc": data.get("risk_fallback", {}).get("image_auroc", 0),
                    "random_auroc": data.get("random_fallback", {}).get("image_auroc", 0),
                }
    return results

def analyze():
    base = r"C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\05_运行记录\本地运行_2026-08-22"
    
    # Load MVTec results
    results = load_mvtec_results(base)
    
    if not results:
        print("No MVTec results found")
        return
    
    fast_vals = [r["fast_auroc"] for r in results.values()]
    risk_vals = [r["risk_auroc"] for r in results.values()]
    random_vals = [r["random_auroc"] for r in results.values()]
    deltas = [r["risk_auroc"] - r["random_auroc"] for r in results.values()]
    
    print("=== 027 MVTec Bootstrap CI Analysis ===")
    print(f"N = {len(results)} units (15 categories × 3 seeds)")
    print()
    
    for name, vals in [("Fast Only", fast_vals), ("Risk Fallback", risk_vals), 
                        ("Random Fallback", random_vals), ("Risk-Random Delta", deltas)]:
        mean = np.mean(vals)
        ci_low, ci_high = bootstrap_ci(vals)
        print(f"{name}: {mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")
    
    # Sign test
    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    print(f"\nSign test: Risk > Random in {n_positive}/{len(deltas)} units")
    print(f"  Positive: {n_positive}, Negative: {n_negative}, Zero: {len(deltas) - n_positive - n_negative}")

if __name__ == "__main__":
    analyze()
