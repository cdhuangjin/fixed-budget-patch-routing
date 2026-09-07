"""027 全部15类MVTec AD评估 - SCI一区标准"""
import subprocess, sys, json
from pathlib import Path

CODE_DIR = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\02_代码")
MVTec_ROOT = r"C:/Users/PC/Documents/Codex/实验/06_主线项目/027_自适应稀疏注意力与准确率效率前沿/04_数据与划分/MVTec AD"
CHECKPOINT_ROOT = r"C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\05_运行记录\seeds_17"
OUTPUT_DIR = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录\027_full_15class")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

print(f"=== 027 全部{len(CATEGORIES)}类MVTec评估 ===")

cmd = [sys.executable, str(CODE_DIR / "evaluate_mvtec_patchcore.py"),
       "--data-root", MVTec_ROOT,
       "--categories"] + CATEGORIES + [
       "--output-root", str(OUTPUT_DIR),
       "--checkpoint-root", CHECKPOINT_ROOT,
       "--device", "cuda"]

print(f"运行命令: {' '.join(cmd[:10])}...")
result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(CODE_DIR))

if result.returncode == 0:
    print("评估完成!")
    print(result.stdout[-500:])
else:
    print(f"错误: {result.stderr[-500:]}")

# 汇总结果
print("\n=== 结果汇总 ===")
for f in sorted(OUTPUT_DIR.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    cat = d.get("category", f.stem)
    print(f"{cat}: {json.dumps({k: v for k, v in d.items() if k not in ('full_only', 'sparse_only', 'random_fallback', 'risk_fallback')})}")
