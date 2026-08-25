import numpy as np, json
from pathlib import Path
from scipy import stats as sp_stats

base = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录")
n_boot = 10000
rng = np.random.default_rng(42)

def analyze_dataset(name, seeds_data):
    risk = np.array([d["risk"] for d in seeds_data])
    random = np.array([d["random"] for d in seeds_data])
    delta = risk - random
    t_stat, p_val = sp_stats.ttest_rel(risk, random)
    cohens_d = delta.mean() / delta.std(ddof=1) if delta.std(ddof=1) > 0 else float("inf")
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(len(delta), size=len(delta), replace=True)
        boot_means.append(delta[idx].mean())
    ci_lo, ci_hi = np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5)
    print(f"\n=== {name} ===")
    print(f"Risk avg:   {risk.mean():.4f} +/- {risk.std(ddof=1):.4f}")
    print(f"Random avg: {random.mean():.4f} +/- {random.std(ddof=1):.4f}")
    print(f"Delta avg:  {delta.mean():.4f} +/- {delta.std(ddof=1):.4f}")
    print(f"Paired t: t={t_stat:.3f}, p={p_val:.6f}")
    print(f"Cohen d:    {cohens_d:.3f}")
    print(f"95% CI:     [{ci_lo:.4f}, {ci_hi:.4f}]")
    return {"risk_mean": float(risk.mean()), "risk_std": float(risk.std(ddof=1)),
            "random_mean": float(random.mean()), "random_std": float(random.std(ddof=1)),
            "delta_mean": float(delta.mean()), "delta_std": float(delta.std(ddof=1)),
            "t_stat": float(t_stat), "p_value": float(p_val), "cohens_d": float(cohens_d),
            "ci_95": [float(ci_lo), float(ci_hi)]}

# VisA
visa_seeds = []
for seed in [5, 17, 29, 41, 53]:
    if seed == 5:
        rf = base / "一区候选_027_visa_phase1" / "seed5" / "results.json"
    else:
        rf = base / "一区候选_027_phase1_multiseed" / "visa" / f"seed{seed}" / "results.json"
    data = json.loads(rf.read_text(encoding="utf-8"))
    visa_seeds.append({"risk": np.mean([r["risk_combined_auroc"] for r in data]),
                       "random": np.mean([r["random_combined_auroc"] for r in data])})

# MPDD
mpdd_seeds = []
for seed in [5, 17, 29, 41, 53]:
    if seed == 5:
        rf = base / "一区候选_027_mpdd_phase1" / "seed5" / "results.json"
    else:
        rf = base / "一区候选_027_phase1_multiseed" / "mpdd" / f"seed{seed}" / "results.json"
    data = json.loads(rf.read_text(encoding="utf-8"))
    mpdd_seeds.append({"risk": np.mean([r["risk_combined_auroc"] for r in data]),
                       "random": np.mean([r["random_combined_auroc"] for r in data])})

visa_r = analyze_dataset("VisA (12 categories, 5 seeds)", visa_seeds)
mpdd_r = analyze_dataset("MPDD (6 categories, 5 seeds)", mpdd_seeds)

out = {"visa": visa_r, "mpdd": mpdd_r}
(base / "一区候选_027_phase1_multiseed" / "statistical_analysis.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nSaved.")
