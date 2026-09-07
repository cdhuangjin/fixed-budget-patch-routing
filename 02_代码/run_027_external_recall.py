#!/usr/bin/env python3
"""Re-run MPDD and VisA to add Recall@5%FPR to the strong-routing raw results."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent

MPDD_CATEGORIES = [
    "bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes",
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
SEEDS = [5, 17, 29]
OUTPUT_ROOT = PROJECT_ROOT / "05_运行记录" / "strong_routing_3seed_v1"


def _run(command: list[str], label: str) -> None:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=CODE_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    print(f"finished {label} in {elapsed:.1f}s rc={completed.returncode}", flush=True)
    if completed.returncode:
        print(completed.stderr[-2000:], flush=True)
        raise SystemExit(f"{label} failed")


def main() -> None:
    for seed in SEEDS:
        mpdd_dir = OUTPUT_ROOT / "mpdd" / f"seed{seed}"
        mpdd_dir.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable, str(CODE_ROOT / "evaluate_mpdd_external.py"),
            "--data-root", str(PROJECT_ROOT / "04_数据与划分" / "MPDD"),
            "--categories", *MPDD_CATEGORIES,
            "--output-root", str(mpdd_dir),
            "--seed", str(seed),
            "--device", "cuda",
            "--fallback-budget", "0.25",
            "--batch-size", "16", "--route-local",
        ], f"mpdd seed={seed}")

        visa_dir = OUTPUT_ROOT / "visa" / f"seed{seed}"
        visa_dir.mkdir(parents=True, exist_ok=True)
        _run([
            sys.executable, str(CODE_ROOT / "evaluate_visa_external.py"),
            "--data-root", str(PROJECT_ROOT / "04_数据与划分" / "VisA_20220922"),
            "--categories", *VISA_CATEGORIES,
            "--output-root", str(visa_dir),
            "--seed", str(seed),
            "--device", "cuda",
            "--fallback-budget", "0.25",
            "--route-local",
        ], f"visa seed={seed}")

    print("ALL EXTERNAL RECALL RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
