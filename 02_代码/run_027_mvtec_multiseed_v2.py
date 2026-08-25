"""Multi-seed MVTec evaluation for 027 using evaluate_mvtec_patchcore.py (correct script)."""
import subprocess, sys, json, statistics
from pathlib import Path

PYTHON = sys.executable
SCRIPT = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\02_代码\evaluate_mvtec_patchcore.py")
DATA = r"C:/Users/PC/Documents/Codex/实验/06_主线项目/027_自适应稀疏注意力与准确率效率前沿/04_数据与划分/MVTec AD"
OUT = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录\一区候选_027_mvtec_multiseed_v2")
OUT.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
              "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
SEEDS = [5, 17, 29]

for seed in SEEDS:
    seed_dir = OUT / f"seed{seed}"
    if seed_dir.exists() and list(seed_dir.glob("*.json")):
        n = len(list(seed_dir.glob("*.json")))
        print(f"SKIP seed={seed} (already has {n} files)")
        continue
    
    cmd = [PYTHON, str(SCRIPT), "--data-root", DATA, "--categories"] + CATEGORIES + [
           "--output-root", str(seed_dir), "--seed", str(seed), "--device", "cuda"]
    print(f"Running seed={seed}...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT.parent))
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
    else:
        print(f"  Done. Output files: {len(list(seed_dir.glob('*.json')))}")

# Summary
print("\n=== MULTI-SEED SUMMARY ===")
for method_name in ("fast_only", "risk_fallback"):
    print(f"\n--- {method_name} ---")
    for seed in SEEDS:
        seed_dir = OUT / f"seed{seed}"
        aurocs = []
        for cat in CATEGORIES:
            f = seed_dir / f"{cat}.json"
            if f.exists():
                d = json.loads(f.read_text(encoding="utf-8"))
                if method_name in d:
                    aurocs.append(d[method_name]["image_auroc"])
        if aurocs:
            print(f"  seed={seed}: mean AUROC={statistics.mean(aurocs):.4f} (n={len(aurocs)})")
