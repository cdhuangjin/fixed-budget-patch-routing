import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from build_strong_routing_budget_canonical import MVTEC_CATEGORIES, _collect, _summarize_one
from materialize_budget_canonical import _audit


def _record(category: str, total: int, budget: float, count: int, auroc_bias: float = 0.0) -> dict:
    route = {
        "actual_fallback_count": count,
        "route_source": "test_free_exact_quota_score_ranking",
    }
    return {
        "category": category,
        "fallback_budget": budget,
        "n_test": total,
        "fast_only": {"image_auroc": 0.9, "recall_at_fpr": 0.5},
        "risk_fallback": {"image_auroc": 0.8 + auroc_bias, "recall_at_fpr": 0.6},
        "strict_quota": {**route, "fallback_rate": count / total},
        "strict_quota_random": {"image_auroc": 0.7, "recall_at_fpr": 0.4, **route},
        "fast_score": {"image_auroc": 0.75, "recall_at_fpr": 0.5, **route},
        "uncertainty_dispersion": {"image_auroc": 0.65, "recall_at_fpr": 0.4, **route},
        "routing": {
            "strict_quota": route,
            "strict_quota_random": route,
            "fast_score": route,
            "uncertainty_dispersion": route,
        },
    }


class StrongRoutingBudgetCanonicalTests(unittest.TestCase):
    def _write(self, root: Path, budget: float, seed: int) -> None:
        total = 32
        count = max(1, round(total * budget))
        for category in MVTEC_CATEGORIES:
            record = _record(category, total, budget, count, auroc_bias=0.05)
            record["seed"] = seed
            path = root / f"seed{seed}" / f"{category}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record), encoding="utf-8")

    def test_collect_requires_full_category_seed_layout(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (5, 17, 29):
                self._write(root, 0.25, seed)
            rows = _collect(root, 0.25)
            self.assertEqual(len(rows), 45)
            self.assertEqual(rows[0]["fallback_count"], 8)
            self.assertAlmostEqual(rows[0]["risk_delta"], 0.15)
            self.assertAlmostEqual(rows[0]["risk_minus_fast_score"], 0.05 + 0.05)
            self.assertAlmostEqual(rows[0]["risk_minus_uncertainty"], 0.15 + 0.05)

            (root / "seed5" / "bottle.json").unlink()
            with self.assertRaises(FileNotFoundError):
                _collect(root, 0.25)

    def test_summarize_paired_bootstrap(self):
        rows = []
        for value in (0.05, 0.06, 0.07):
            rows.append({
                "risk_auroc": 0.8 + value,
                "random_auroc": 0.7,
                "risk_recall": 0.6,
                "random_recall": 0.4,
                "fast_score_auroc": 0.75,
                "uncertainty_auroc": 0.65,
            })
        summary = _summarize_one(rows)
        self.assertEqual(summary["unit_count"], 3)
        self.assertGreater(summary["auroc_delta_mean"], 0.05)
        self.assertGreater(summary["recall_delta_mean"], 0.1)
        self.assertGreater(summary["auroc_positive_count"], 0)

    def test_materialize_audit_seed_layout_uses_fallback_budget_label(self):
        rows = []
        for budget in (0.10, 0.25, 0.50):
            for seed in (5, 17, 29):
                for category in MVTEC_CATEGORIES:
                    total = 32
                    count = math.ceil(total * budget - 1e-12)
                    rows.append({
                        "category": category, "seed": seed, "fallback_budget": budget,
                        "fallback_rate": count / total, "fallback_count": count, "total": total,
                        "risk_auroc": 0.85, "random_auroc": 0.70, "fast_score_auroc": 0.75,
                        "uncertainty_auroc": 0.65, "risk_recall": 0.60, "random_recall": 0.40,
                        "risk_delta": 0.15, "risk_recall_delta": 0.20,
                        "risk_minus_fast_score": 0.10, "risk_minus_uncertainty": 0.20,
                    })
        canonical = {
            "protocol": {"fallback_budgets": [0.10, 0.25, 0.50], "seeds": [5, 17, 29]},
            "rows": rows,
        }
        audit = _audit(canonical)
        checks = {check["name"]: check["status"] for check in audit["checks"]}
        self.assertEqual(checks["seed_layout"], "PASS")
        self.assertEqual(checks["budget_row_counts"], "PASS")
        self.assertEqual(audit["summary"]["fail"], 0)


if __name__ == "__main__":
    unittest.main()
