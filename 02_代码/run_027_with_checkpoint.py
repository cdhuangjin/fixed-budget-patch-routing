"""027项目完整实验 - 使用checkpoint"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# GPU优化
sys.path.insert(0, r"C:\Users\PC\Documents\Codex\实验\04_脚本工具")
try:
    from gpu_optimize import setup_rtx5060_optimizations
    setup_rtx5060_optimizations()
except ImportError:
    pass

# 路径配置
MVTec_ROOT = r"C:/Users/PC/Documents/Codex/实验/06_主线项目/027_自适应稀疏注意力与准确率效率前沿/04_数据与划分/MVTec AD"
VisA_ROOT = r"C:/Users/PC/Documents/Codex/实验/06_主线项目/027_自适应稀疏注意力与准确率效率前沿/04_数据与划分\VisA/extracted/VisA_20220922"
CHECKPOINT_ROOT = r"C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\05_运行记录\seeds_17"
CODE_DIR = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\02_代码")
OUTPUT_DIR = Path(CODE_DIR).parent / "05_运行记录" / f"027_完整实验_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, cwd=None):
    print(f"运行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"错误: {result.stderr[:500]}")
    else:
        print(f"完成")
    return result

# MVTec评估
print("=" * 60)
print("027项目完整实验开始")
print("=" * 60)

categories = ["bottle", "cable", "capsule", "carpet", "grid"]
print(f"运行MVTec AD评估（{len(categories)}类别）...")
cmd = f'python "{CODE_DIR}/evaluate_mvtec_selective.py" --data-root "{MVTec_ROOT}" --categories {" ".join(categories)} --output-root "{OUTPUT_DIR}/mvtec" --fallback-budget 0.25 --checkpoint-root "{CHECKPOINT_ROOT}"'
run_cmd(cmd, cwd=str(CODE_DIR))

# VisA外部验证
print("运行VisA外部验证...")
visa_categories = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]
cmd = f'python "{CODE_DIR}/evaluate_visa_external.py" --data-root "{VisA_ROOT}" --categories {" ".join(visa_categories)} --output-root "{OUTPUT_DIR}/visa" --seed 17 --fallback-budget 0.25'
run_cmd(cmd, cwd=str(CODE_DIR))

# 延迟基准测试
print("运行延迟基准测试...")
cmd = f'python "{CODE_DIR}/benchmark_latency.py" --output "{OUTPUT_DIR}/latency.json"'
run_cmd(cmd, cwd=str(CODE_DIR))

print("=" * 60)
print(f"完成，结果保存在: {OUTPUT_DIR}")
print("=" * 60)
