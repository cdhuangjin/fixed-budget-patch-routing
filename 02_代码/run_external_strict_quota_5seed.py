#!/usr/bin/env python3
"""Re-run MPDD and VisA under the label-free, exact-quota 027 protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
MPDD_CATEGORIES = [
    "bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes",
]
SEEDS = [5, 17, 29, 41, 53]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--mpdd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fallback-budget", type=float, default=0.25)
    args = parser.parse_args()
    code_root = Path(__file__).parent
    jobs = [
        ("visa", "evaluate_visa_external.py", args.visa_root, VISA_CATEGORIES),
        ("mpdd", "evaluate_mpdd_external.py", args.mpdd_root, MPDD_CATEGORIES),
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_log = []
    for seed in SEEDS:
        for dataset, script, data_root, categories in jobs:
            output_dir = args.output_root / dataset / f"seed{seed}"
            output_file = output_dir / "results.json"
            if output_file.exists():
                run_log.append({"dataset": dataset, "seed": seed, "status": "skipped_existing"})
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(code_root / script), "--data-root", str(data_root),
                "--categories", *categories, "--output-root", str(output_dir),
                "--seed", str(seed), "--device", args.device,
                "--fallback-budget", str(args.fallback_budget),
            ]
            if dataset == "mpdd":
                command.extend(["--batch-size", "16"])
            started = time.perf_counter()
            completed = subprocess.run(command, cwd=code_root, capture_output=True, text=True)
            elapsed = time.perf_counter() - started
            (output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            record = {
                "dataset": dataset, "seed": seed, "returncode": completed.returncode,
                "elapsed_seconds": elapsed, "result_exists": output_file.exists(),
            }
            run_log.append(record)
            (args.output_root / "run_log.json").write_text(
                json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if completed.returncode:
                raise SystemExit(f"{dataset} seed={seed} failed; see {output_dir / 'stderr.log'}")


if __name__ == "__main__":
    main()
