#!/usr/bin/env python3
"""027 Phase1 multiseed: seed 17/29/41/53 on VisA + MPDD"""
import subprocess, sys, json
from datetime import datetime
from pathlib import Path

PYTHON = sys.executable
CODE_DIR = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目/027_自适应稀疏注意力与准确率效率前沿\02_代码")
OUTPUT_ROOT = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录\一区候选_027_phase1_multiseed")

VISA_DATA = r"C:/Users/PC/Documents/Codex/实验/06_主线项目/027_自适应稀疏注意力与准确率效率前沿/04_数据与划分/VisA/extracted"
MPDD_DATA = r"C:/Users/PC/Documents/Codex/实验/06_主线项目/027_自适应稀疏注意力与准确率效率前沿/04_数据与划分\MPDD/extracted/MPDD"
VISA_CATS = ["candle","capsules","cashew","chewinggum","fryum","macaroni1","macaroni2","pcb1","pcb2","pcb3","pcb4","pipe_fryum"]
MPDD_CATS = ["bracket_black","bracket_brown","bracket_white","connector","metal_plate","tubes"]
SEEDS = [17, 29, 41, 53]

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
all_results = []

for seed in SEEDS:
    for dataset, cats, script, data_root in [
        ("visa", VISA_CATS, "evaluate_visa_external.py", VISA_DATA),
        ("mpdd", MPDD_CATS, "evaluate_mpdd_external.py", MPDD_DATA),
    ]:
        tag = f"{dataset}_seed{seed}"
        out_dir = OUTPUT_ROOT / dataset / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [PYTHON, str(CODE_DIR / script),
               "--data-root", data_root,
               "--categories"] + cats + [
               "--output-root", str(out_dir),
               "--seed", str(seed), "--device", "cuda"]
        if dataset == "mpdd":
            cmd += ["--batch-size", "16"]
        print(f"[{tag}] starting...", flush=True)
        t0 = datetime.now()
        try:
            r = subprocess.run(cmd, cwd=str(CODE_DIR), capture_output=True, text=True, timeout=3600)
            rc = r.returncode
            err_tail = (r.stderr or "")[-300:]
        except Exception as e:
            rc = -1
            err_tail = repr(e)
        dt = (datetime.now() - t0).total_seconds()
        ok = "OK" if rc == 0 else "FAIL"
        print(f"[{tag}] {ok} in {dt:.0f}s", flush=True)
        if rc != 0:
            print(f"  err: {err_tail[:200]}", flush=True)
        all_results.append({"tag": tag, "seed": seed, "dataset": dataset, "rc": rc, "time_s": round(dt,1)})

# summary
(OUTPUT_ROOT / "run_log.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

# aggregate
print("\n=== MULTISEED SUMMARY ===")
for dataset in ["visa", "mpdd"]:
    print(f"\n--- {dataset.upper()} ---")
    for seed in [5] + SEEDS:
        if seed == 5:
            rf = Path(r"C:\Users\PC\Documents\Codex\实验\06_主线项目\027_自适应稀疏注意力与准确率效率前沿\05_运行记录") / f"一区候选_027_{dataset}_phase1" / "seed5" / "results.json"
        else:
            rf = OUTPUT_ROOT / dataset / f"seed{seed}" / "results.json"
        if rf.exists():
            data = json.loads(rf.read_text(encoding="utf-8"))
            fa = sum(r["fast_only_auroc"] for r in data)/len(data)
            ra = sum(r["risk_combined_auroc"] for r in data)/len(data)
            aa = sum(r["random_combined_auroc"] for r in data)/len(data)
            d = ra - aa
            gc = sum(1 for r in data if r["risk_delta"] > 0)
            print(f"  seed={seed}: fast={fa:.4f} risk={ra:.4f} random={aa:.4f} delta={d:+.4f} ({gc}/{len(data)})")
        else:
            print(f"  seed={seed}: NO DATA")
