#!/usr/bin/env python3
"""Build an auditable, matched-budget 027 strict-quota canonical table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_canonical_027_main_table import collect_mvtec, summarize


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_external_strict_quota(root: Path) -> list[dict]:
    """Load external results and reject every route that is not exactly matched."""
    rows: list[dict] = []
    name_map = {"visa": "VisA", "mpdd": "MPDD"}
    for disk_name, dataset in name_map.items():
        for path in sorted((root / disk_name).glob("seed*/results.json")):
            for item in _read(path):
                route, random_route = item["route"], item["random_route"]
                risk_count = int(route["actual_fallback_count"])
                random_count = int(random_route["actual_fallback_count"])
                if route.get("route_name") != "strict_quota":
                    raise ValueError(f"{path}: route is not strict_quota")
                if random_route.get("route_name") != "random_matched":
                    raise ValueError(f"{path}: random route is not matched")
                if risk_count != random_count:
                    raise ValueError(f"{path}: matched random quota differs from risk quota")
                if risk_count != int(item["risk_count"]):
                    raise ValueError(f"{path}: recorded risk count differs from route")
                if abs(float(route["actual_rate"]) - float(item["fallback_rate"])) > 1e-12:
                    raise ValueError(f"{path}: recorded fallback rate differs from route")
                rows.append({
                    "dataset": dataset,
                    "category": item["category"],
                    "seed": int(item["seed"]),
                    "fallback_budget": float(item["budget"]),
                    "fast_only_auroc": float(item["fast_only_auroc"]),
                    "risk_auroc": float(item["risk_combined_auroc"]),
                    "random_auroc": float(item["random_combined_auroc"]),
                    "risk_delta": float(item["risk_delta"]),
                    "fallback_rate": float(item["fallback_rate"]),
                    "risk_count": risk_count,
                    "random_count": random_count,
                    "total": int(item["total"]),
                    "route_source": route.get("route_source"),
                })
    if not rows:
        raise ValueError(f"no external strict-quota rows found under {root}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvtec-dir", required=True)
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = collect_mvtec(Path(args.mvtec_dir), route="strict_quota")
    for row in rows:
        required = ("risk_count", "random_count", "total", "route_source")
        if any(key not in row for key in required):
            raise ValueError("MVTec strict-quota source lacks routing provenance")
        if row["risk_count"] != row["random_count"]:
            raise ValueError("MVTec strict-quota source has unmatched fallback counts")
        if abs(row["fallback_rate"] - row["risk_count"] / row["total"]) > 1e-12:
            raise ValueError("MVTec strict-quota source has inconsistent fallback rate")
    rows.extend(collect_external_strict_quota(Path(args.external_dir)))
    canonical = {
        "status": "canonical_strict_quota_main_table",
        "task": "test_free_transductive_selective_inference",
        "protocol": {
            "fallback_budget": 0.25,
            "route": "strict_quota",
            "random_control": "same per-category fallback count, seeded uniform sampling",
            "scope": "test-free global score ranking; not an online-prefix guarantee",
        },
        "rows": rows,
        "summary": summarize(rows),
        "audit_notes": [
            "Every external row is rejected unless risk and random use identical fallback counts.",
            "MVTec rows come from the same strict_quota route; external data are VisA and MPDD, each with five seeds.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(canonical, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "summary": canonical["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
