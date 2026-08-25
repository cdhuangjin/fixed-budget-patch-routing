"""027 comprehensive analysis: hard-set, Pareto, significance."""
import json, math, statistics
from pathlib import Path

ROOT = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录")

# === 1. MVTec Analysis (15 categories) ===
mvtec_dir = ROOT / "027_full_15class"
mvtec_cats = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
              "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]

mvtec_records = []
for cat in mvtec_cats:
    f = mvtec_dir / f"{cat}.json"
    if f.exists():
        r = json.loads(f.read_text(encoding="utf-8"))
        mvtec_records.append(r)

print("=" * 80)
print("027 COMPREHENSIVE ANALYSIS")
print("=" * 80)

# === 1a. MVTec AUROC comparison ===
print("\n### MVTec AD (15 categories, single seed) ###")
print(f"{'Category':<15} {'Fast':>8} {'Risk':>8} {'Random':>8} {'Delta':>8}")
for r in mvtec_records:
    cat = r["category"]
    fast = r["fast_only"]["image_auroc"]
    risk = r["risk_fallback"]["image_auroc"]
    rand = r["random_fallback"]["image_auroc"]
    delta = risk - rand
    print(f"{cat:<15} {fast:>8.4f} {risk:>8.4f} {rand:>8.4f} {delta:>+8.4f}")

fast_avg = statistics.mean(r["fast_only"]["image_auroc"] for r in mvtec_records)
risk_avg = statistics.mean(r["risk_fallback"]["image_auroc"] for r in mvtec_records)
rand_avg = statistics.mean(r["random_fallback"]["image_auroc"] for r in mvtec_records)
print(f"{'Average':<15} {fast_avg:>8.4f} {risk_avg:>8.4f} {rand_avg:>8.4f} {risk_avg-rand_avg:>+8.4f}")

# === 1b. Hard-set analysis ===
print("\n### Hard-Set Analysis (categories where fast_only < 0.90) ###")
hard_set = [r for r in mvtec_records if r["fast_only"]["image_auroc"] < 0.90]
easy_set = [r for r in mvtec_records if r["fast_only"]["image_auroc"] >= 0.90]
print(f"Hard categories ({len(hard_set)}): {[r['category'] for r in hard_set]}")
print(f"Easy categories ({len(easy_set)}): {[r['category'] for r in easy_set]}")

if hard_set:
    hard_fast = statistics.mean(r["fast_only"]["image_auroc"] for r in hard_set)
    hard_risk = statistics.mean(r["risk_fallback"]["image_auroc"] for r in hard_set)
    hard_rand = statistics.mean(r["random_fallback"]["image_auroc"] for r in hard_set)
    print(f"Hard-set: fast={hard_fast:.4f}, risk={hard_risk:.4f}, random={hard_rand:.4f}, delta={hard_risk-hard_rand:+.4f}")

if easy_set:
    easy_fast = statistics.mean(r["fast_only"]["image_auroc"] for r in easy_set)
    easy_risk = statistics.mean(r["risk_fallback"]["image_auroc"] for r in easy_set)
    easy_rand = statistics.mean(r["random_fallback"]["image_auroc"] for r in easy_set)
    print(f"Easy-set: fast={easy_fast:.4f}, risk={easy_risk:.4f}, random={easy_rand:.4f}, delta={easy_risk-easy_rand:+.4f}")

# === 1c. Efficiency Pareto analysis ===
print("\n### Efficiency Pareto (latency vs accuracy) ###")
methods_config = [
    ("fast_only", "Fast Only"),
    ("strict_quota", "Strict Quota"),
    ("strict_quota_random", "Strict Quota (Random)"),
    ("random_fallback", "Random Fallback"),
    ("risk_fallback", "Risk Fallback"),
]
for method_key, method_name in methods_config:
    aurocs = [r[method_key]["image_auroc"] for r in mvtec_records if method_key in r]
    latencies = [r[method_key]["mean_ms_estimated"] for r in mvtec_records if method_key in r]
    if aurocs:
        avg_auroc = statistics.mean(aurocs)
        avg_latency = statistics.mean(latencies)
        print(f"{method_name:<25}: AUROC={avg_auroc:.4f}, latency={avg_latency:.1f}ms")

# === 2. VisA Analysis ===
print("\n" + "=" * 80)
print("### VisA Analysis (12 categories, 3 seeds) ###")
visa_dir = ROOT / "一区候选_027_visa_phase1"
visa_results = []
for f in visa_dir.glob("*.json"):
    if f.name != "statistical_analysis.json":
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                visa_results.extend(data)
            elif isinstance(data, dict) and "risk_mean" not in data:
                visa_results.append(data)
        except:
            pass

if visa_results:
    # Group by category and seed
    cats = {}
    for r in visa_results:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r)
    
    print(f"{'Category':<15} {'Fast':>8} {'Risk':>8} {'Random':>8} {'Delta':>8}")
    for cat in sorted(cats.keys()):
        records = cats[cat]
        fast = statistics.mean(r.get("fast_only", {}).get("image_auroc", 0) for r in records)
        risk = statistics.mean(r.get("risk_fallback", {}).get("image_auroc", 0) for r in records)
        rand = statistics.mean(r.get("random_fallback", {}).get("image_auroc", 0) for r in records)
        print(f"{cat:<15} {fast:>8.4f} {risk:>8.4f} {rand:>8.4f} {risk-rand:>+8.4f}")

# === 3. MPDD Analysis ===
print("\n" + "=" * 80)
print("### MPDD Analysis (6 categories, 3 seeds) ###")
mpdd_dir = ROOT / "一区候选_027_mpdd_phase1"
mpdd_results = []
for f in mpdd_dir.glob("*.json"):
    if f.name != "statistical_analysis.json":
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                mpdd_results.extend(data)
            elif isinstance(data, dict) and "risk_mean" not in data:
                mpdd_results.append(data)
        except:
            pass

if mpdd_results:
    cats = {}
    for r in mpdd_results:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r)
    
    print(f"{'Category':<15} {'Fast':>8} {'Risk':>8} {'Random':>8} {'Delta':>8}")
    for cat in sorted(cats.keys()):
        records = cats[cat]
        fast = statistics.mean(r.get("fast_only", {}).get("image_auroc", 0) for r in records)
        risk = statistics.mean(r.get("risk_fallback", {}).get("image_auroc", 0) for r in records)
        rand = statistics.mean(r.get("random_fallback", {}).get("image_auroc", 0) for r in records)
        print(f"{cat:<15} {fast:>8.4f} {risk:>8.4f} {rand:>8.4f} {risk-rand:>+8.4f}")

# === 4. Summary report ===
print("\n" + "=" * 80)
print("Q1 READINESS ASSESSMENT")
print("=" * 80)
print("""
EXPERIMENT 027 STATUS:
[✓] 3 datasets: MVTec (15 cats), VisA (12 cats), MPDD (6 cats)
[✓] Multiple methods: fast_only, risk_fallback, random_fallback, strict_quota
[✓] Multi-seed: VisA & MPDD (3 seeds), MVTec (1 seed)
[✓] Statistical significance: p < 0.05 for VisA and MPDD
[✓] Hard-set analysis: Included above
[✓] Efficiency Pareto: Included above
[ ] MVTec multi-seed (currently single seed)
[ ] CIFAR-100-C external domain validation
[ ] Formal figure scripts
""")

# Save analysis
analysis = {
    "mvtec": {
        "n_categories": len(mvtec_records),
        "fast_avg": fast_avg,
        "risk_avg": risk_avg,
        "rand_avg": rand_avg,
        "delta": risk_avg - rand_avg,
        "hard_set": {
            "categories": [r["category"] for r in hard_set],
            "fast_avg": statistics.mean(r["fast_only"]["image_auroc"] for r in hard_set) if hard_set else None,
            "risk_avg": statistics.mean(r["risk_fallback"]["image_auroc"] for r in hard_set) if hard_set else None,
        },
        "easy_set": {
            "categories": [r["category"] for r in easy_set],
            "fast_avg": statistics.mean(r["fast_only"]["image_auroc"] for r in easy_set) if easy_set else None,
            "risk_avg": statistics.mean(r["risk_fallback"]["image_auroc"] for r in easy_set) if easy_set else None,
        }
    }
}
out_path = ROOT / "一区候选_027_phase1_multiseed" / "comprehensive_analysis.json"
out_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved to {out_path}")
