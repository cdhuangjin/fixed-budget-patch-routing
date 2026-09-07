#!/usr/bin/env python3
"""Run the plan-mandated strong routing baselines with three formal seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
MPDD_CATEGORIES = [
    "bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes",
]
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
SEEDS = [5, 17, 29]


def append_log(output_root: Path, record: dict) -> None:
    log_path = output_root / "run_log.json"
    records = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    records.append(record)
    log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str], output_root: Path, dataset: str, seed: int) -> None:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=CODE_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    record = {
        "dataset": dataset,
        "seed": seed,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
    }
    append_log(output_root, record)
    if completed.returncode:
        (output_root / f"{dataset}_seed{seed}_stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        raise SystemExit(f"{dataset} seed={seed} failed; returncode={completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "05_运行记录" / "strong_routing_3seed_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fallback-budget", type=float, default=0.25)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        mvtec_output = args.output_root / "mvtec" / f"seed{seed}"
        mvtec_output.mkdir(parents=True, exist_ok=True)
        for category in MVTEC_CATEGORIES:
            output_file = mvtec_output / f"{category}.json"
            if output_file.exists():
                continue
            run(
                [
                    sys.executable, str(CODE_ROOT / "evaluate_mvtec_patchcore.py"),
                    "--data-root", str(PROJECT_ROOT / "04_数据与划分" / "MVTec AD"),
                    "--categories", category,
                    "--output-root", str(mvtec_output),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--fallback-budget", str(args.fallback_budget),
                    "--route-local", "--local-top-fraction", "0.05",
                ],
                args.output_root, "mvtec", seed,
            )

        mpdd_output = args.output_root / "mpdd" / f"seed{seed}" / "results.json"
        if not mpdd_output.exists():
            run(
                [
                    sys.executable, str(CODE_ROOT / "evaluate_mpdd_external.py"),
                    "--data-root", str(PROJECT_ROOT / "04_数据与划分" / "MPDD"),
                    "--categories", *MPDD_CATEGORIES,
                    "--output-root", str(mpdd_output.parent),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--fallback-budget", str(args.fallback_budget),
                    "--batch-size", "16", "--route-local",
                ],
                args.output_root, "mpdd", seed,
            )

        visa_output = args.output_root / "visa" / f"seed{seed}" / "results.json"
        if not visa_output.exists():
            run(
                [
                    sys.executable, str(CODE_ROOT / "evaluate_visa_external.py"),
                    "--data-root", str(PROJECT_ROOT / "04_数据与划分" / "VisA_20220922"),
                    "--categories", *VISA_CATEGORIES,
                    "--output-root", str(visa_output.parent),
                    "--seed", str(seed),
                    "--device", args.device,
                    "--fallback-budget", str(args.fallback_budget),
                    "--route-local",
                ],
                args.output_root, "visa", seed,
            )


if __name__ == "__main__":
    main()
