import json, numpy as np
from pathlib import Path
from scipy import stats as sp_stats

base = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录")

# === 1. Failure case analysis: which categories have risk < random? ===
print("=== 027 FAILURE CASE ANALYSIS ===")
for ds_name, ds_path in [("VisA", "一区候选_027_visa_phase1"), ("MPDD", "一区候选_027_mpdd_phase1")]:
    print(f"\n--- {ds_name} ---")
    # Aggregate across all seeds
    all_cat_deltas = {}
    for seed in [5, 17, 29, 41, 53]:
        if seed == 5:
            rf = base / ds_path / "seed5" / "results.json"
        else:
            rf = base / "一区候选_027_phase1_multiseed" / ds_name.lower() / f"seed{seed}" / "results.json"
        if rf.exists():
            data = json.loads(rf.read_text(encoding="utf-8"))
            for r in data:
                cat = r["category"]
                if cat not in all_cat_deltas:
                    all_cat_deltas[cat] = []
                all_cat_deltas[cat].append(r["risk_delta"])
    
    # Per-category statistics
    weak_cats = []
    for cat in sorted(all_cat_deltas):
        deltas = np.array(all_cat_deltas[cat])
        mean_d = deltas.mean()
        if mean_d < 0:
            weak_cats.append((cat, mean_d))
        print(f"  {cat:20s}: delta={mean_d:+.4f} (n={len(deltas)})")
    
    if weak_cats:
        print(f"  ** Weak categories (risk < random): {weak_cats}")
    else:
        print(f"  ** All categories have risk >= random")

# === 2. Budget sensitivity: check existing data for different budgets ===
print("\n=== 027 BUDGET SENSITIVITY ===")
# Check the original 15-class MVTec results which may have different budgets
mvtec_dir = base / "027_full_15class"
if mvtec_dir.exists():
    for f in sorted(mvtec_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(d, dict) and "budget" in d:
            cat = d.get("category", f.stem)
            budget = d.get("budget", "?")
            fast = d.get("fast_only_auroc", 0)
            risk = d.get("risk_combined_auroc", 0)
            delta = d.get("risk_delta", 0)
            print(f"  {cat}: budget={budget} fast={fast:.4f} risk={risk:.4f} delta={delta:+.4f}")

# === 3. Bootstrap CI for per-category deltas ===
print("\n=== 027 PER-CATEGORY BOOTSTRAP (VisA) ===")
for seed in [5]:
    rf = base / "一区候选_027_visa_phase1" / "seed5" / "results.json"
    data = json.loads(rf.read_text(encoding="utf-8"))
    for r in data:
        cat = r["category"]
        # Single-seed, single-category: just report the delta
        print(f"  {cat:20s}: fast={r['fast_only_auroc']:.4f} risk={r['risk_combined_auroc']:.4f} delta={r['risk_delta']:+.4f}")
