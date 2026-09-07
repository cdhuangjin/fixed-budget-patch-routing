"""Multi-seed MVTec evaluation for 027 - fixed."""
import subprocess, sys, json, statistics
from pathlib import Path

PYTHON = sys.executable
SCRIPT = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\02_代码\evaluate_mvtec_selective.py")
DATA = r"C:/Users/PC/Documents/Codex/实验/06_主线项目/027_自适应稀疏注意力与准确率效率前沿/04_数据与划分/MVTec AD"
OUT = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录\一区候选_027_mvtec_multiseed")
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
              "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
SEEDS = [5, 17, 29]

all_results = []
for seed in SEEDS:
    seed_dir = OUT / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = seed_dir / "results.json"
    if result_file.exists():
        results = json.loads(result_file.read_text(encoding="utf-8"))
        for r in results:
            r["seed"] = seed
            all_results.append(r)
        print(f"SKIP seed={seed} (already exists)")
        continue
    
    cmd = [PYTHON, str(SCRIPT), "--data-root", DATA, "--categories"] + CATEGORIES + [
           "--output-root", str(seed_dir), "--seed", str(seed), "--device", "cuda"]
    print(f"Running seed={seed}...", flush=True)
    subprocess.run(cmd, check=True)
    
    if result_file.exists():
        results = json.loads(result_file.read_text(encoding="utf-8"))
        for r in results:
            r["seed"] = seed
            all_results.append(r)

# Summary
print("\n=== MULTI-SEED SUMMARY ===")
cats_data = {}
for r in all_results:
    cat = r["category"]
    cats_data.setdefault(cat, []).append(r)

print(f"{'Category':<15} {'Risk':>8} {'Random':>8} {'Delta':>8} {'n_seeds':>8}")
for cat in CATEGORIES:
    if cat in cats_data:
        risk_vals = [r["risk_combined_auroc"] for r in cats_data[cat]]
        rand_vals = [r["random_combined_auroc"] for r in cats_data[cat]]
        risk_avg = statistics.mean(risk_vals)
        rand_avg = statistics.mean(rand_vals)
        risk_std = statistics.pstdev(risk_vals) if len(risk_vals) > 1 else 0
        print(f"{cat:<15} {risk_avg:>8.4f} {rand_avg:>8.4f} {risk_avg-rand_avg:>+8.4f} {len(risk_vals):>8}")

overall_risk = statistics.mean(r["risk_combined_auroc"] for r in all_results)
overall_rand = statistics.mean(r["random_combined_auroc"] for r in all_results)
print(f"{'Overall':<15} {overall_risk:>8.4f} {overall_rand:>8.4f} {overall_risk-overall_rand:>+8.4f}")

# Save all results
out_file = OUT / "all_results.json"
out_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
