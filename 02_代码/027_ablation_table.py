import json, numpy as np
from pathlib import Path

base = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录")

print("=== 027 ABLATION TABLE: Fast vs Risk vs Random ===")
for ds_name, ds_path in [("VisA", "一区候选_027_visa_phase1"), ("MPDD", "一区候选_027_mpdd_phase1")]:
    print(f"\n--- {ds_name} (5 seeds averaged) ---")
    fast_all = []
    risk_all = []
    rand_all = []
    for seed in [5, 17, 29, 41, 53]:
        if seed == 5:
            rf = base / ds_path / "seed5" / "results.json"
        else:
            rf = base / "一区候选_027_phase1_multiseed" / ds_name.lower() / f"seed{seed}" / "results.json"
        if rf.exists():
            data = json.loads(rf.read_text(encoding="utf-8"))
            fast_all.append(np.mean([r["fast_only_auroc"] for r in data]))
            risk_all.append(np.mean([r["risk_combined_auroc"] for r in data]))
            rand_all.append(np.mean([r["random_combined_auroc"] for r in data]))
    
    print(f"  Fast-only:  {np.mean(fast_all):.4f} +/- {np.std(fast_all, ddof=1):.4f}")
    print(f"  Risk-based: {np.mean(risk_all):.4f} +/- {np.std(risk_all, ddof=1):.4f}")
    print(f"  Random:     {np.mean(rand_all):.4f} +/- {np.std(rand_all, ddof=1):.4f}")
    print(f"  Risk-Fast:  {np.mean(risk_all)-np.mean(fast_all):+.4f}")
    print(f"  Risk-Random:{np.mean(risk_all)-np.mean(rand_all):+.4f}")

# Efficiency summary
print("\n=== 027 EFFICIENCY ===")
lat = json.loads(Path(base / "一区候选_027_efficiency" / "latency.json").read_text(encoding="utf-8"))
print(f"  Fast path latency: {lat['fast']['mean_ms']:.2f}ms (p50={lat['fast']['p50_ms']:.2f}ms)")
print(f"  Full path latency: {lat['full']['mean_ms']:.2f}ms (p50={lat['full']['p50_ms']:.2f}ms)")
print(f"  Speedup (full/fast): {lat['full']['mean_ms']/lat['fast']['mean_ms']:.1f}x")
