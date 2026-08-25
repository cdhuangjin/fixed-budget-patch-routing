from pathlib import Path
import json

import numpy as np
import pandas as pd

from analysis_protocol import paired_bootstrap_delta_ci


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "05_运行记录"
OUT = ROOT / "07_论文" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [17, 29, 41]
METHODS = ["fast_only", "random_fallback", "risk_fallback", "full_only"]


def read_units():
    rows = []
    for seed in SEEDS:
        directory = RUNS / f"mvtec_route_ablation_gpu_seed{seed}"
        files = sorted(directory.glob("*.json"))
        if len(files) != 15:
            raise RuntimeError(f"Expected 15 category files for seed {seed}, found {len(files)}")
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            category = payload["category"]
            risk_route = payload["routing"]["risk"]
            random_route = payload["routing"]["random"]
            if risk_route["actual_fallback_count"] != random_route["actual_fallback_count"]:
                raise RuntimeError(f"Fallback mismatch: {seed} {category}")
            row = {
                "seed": seed,
                "category": category,
                "n_fit": payload["n_fit"],
                "n_validation": payload["n_validation"],
                "n_test": payload["n_test"],
                "target_fallback_count": risk_route["target_fallback_count"],
                "actual_fallback_count": risk_route["actual_fallback_count"],
                "actual_fallback_rate": risk_route["actual_rate"],
                "threshold": payload["threshold"],
            }
            for method in METHODS:
                result = payload[method]
                row[f"{method}_auroc"] = result["image_auroc"]
                row[f"{method}_recall_at_fpr"] = result["recall_at_fpr"]
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["seed", "category"]).reset_index(drop=True)
    if len(frame) != 45:
        raise RuntimeError(f"Expected 45 category-seed units, found {len(frame)}")
    return frame


def paired_bootstrap(frame, metric, reps=10000, seed=17):
    left = frame[f"risk_fallback_{metric}"].to_numpy()
    right = frame[f"random_fallback_{metric}"].to_numpy()
    result = paired_bootstrap_delta_ci(left, right, reps=reps, seed=seed)
    delta = left - right
    return {
        "estimate": result["mean"],
        "ci_low": result["low"],
        "ci_high": result["high"],
        "positive_units": int((delta > 0).sum()),
        "n_units": int(len(delta)),
        "reps": result["reps"],
        "seed": seed,
    }


if __name__ == "__main__":
    frame = read_units()
    csv_path = OUT / "mvtec_category_seed_unit_results.csv"
    frame.to_csv(csv_path, index=False)
    summary = {
        "n_units": len(frame),
        "fallback_counts_match_all": bool(
            (frame["target_fallback_count"] == frame["actual_fallback_count"]).all()
        ),
        "risk_random_fallback_counts_match_all": True,
        "risk_vs_random": {
            "auroc": paired_bootstrap(frame, "auroc"),
            "recall_at_fpr": paired_bootstrap(frame, "recall_at_fpr"),
        },
    }
    summary_path = OUT / "mvtec_category_seed_unit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
