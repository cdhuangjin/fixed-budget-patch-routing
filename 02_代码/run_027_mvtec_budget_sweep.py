#!/usr/bin/env python3
"""Run the MVTec 10%/50% three-seed budget sweep for the V3 canonical budget curve.

This is the remaining accepted-budget sensitivity gap: the 25% point is already
three-seed canonical, while 10% and 50% were previously single-seed exploratory.
The script invokes the same MVTec evaluator used by the 25% strong-routing run,
with the same routing flags and the same seeds, so the 10%/25%/50% curve shares
one protocol.

Output layout (matches the dispatcher's per-seed dirs but adds a budget level):
    <output_root>/mvtec/budget_<budget>/seed<seed>/<category>.json
"""

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
SEEDS = [5, 17, 29]
BUDGETS = [0.10, 0.50]


def append_log(output_root: Path, record: dict) -> None:
    log_path = output_root / "run_log.json"
    records = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    records.append(record)
    log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _existing_categories(output_root: Path, budget: float, seed: int) -> set[str]:
    seed_dir = output_root / "mvtec" / f"budget_{budget:.2f}" / f"seed{seed}"
    if not seed_dir.exists():
        return set()
    return {path.stem for path in seed_dir.glob("*.json")}


def run_budget(
    output_root: Path,
    budget: float,
    seed: int,
    device: str,
    data_root: Path,
) -> None:
    budget_dir = output_root / "mvtec" / f"budget_{budget:.2f}"
    seed_dir = budget_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        category for category in MVTEC_CATEGORIES
        if not (seed_dir / f"{category}.json").exists()
    ]
    if not missing:
        print(f"budget={budget} seed={seed}: all {len(MVTEC_CATEGORIES)} categories present; skip")
        return

    categories = missing if missing else MVTEC_CATEGORIES
    command = [
        sys.executable, str(CODE_ROOT / "evaluate_mvtec_patchcore.py"),
        "--data-root", str(data_root),
        "--categories", *categories,
        "--output-root", str(seed_dir),
        "--seed", str(seed),
        "--device", device,
        "--fallback-budget", str(budget),
        "--route-local", "--local-top-fraction", "0.05",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=CODE_ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    record = {
        "budget": budget,
        "seed": seed,
        "categories": categories,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
    }
    append_log(output_root, record)
    if completed.returncode:
        (output_root / f"mvtec_budget{budget:.2f}_seed{seed}_stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        raise SystemExit(
            f"budget={budget} seed={seed} failed; returncode={completed.returncode}\n"
            f"{completed.stderr[-4000:]}"
        )
    print(f"budget={budget} seed={seed}: completed {len(categories)} categories in {elapsed:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "05_运行记录" / "strong_routing_budget_sweep_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "04_数据与划分" / "MVTec AD",
    )
    parser.add_argument("--budgets", nargs="+", type=float, default=BUDGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for budget in args.budgets:
        for seed in args.seeds:
            run_budget(args.output_root, budget, seed, args.device, args.data_root)


if __name__ == "__main__":
    main()
