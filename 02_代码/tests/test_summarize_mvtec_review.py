import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from summarize_mvtec_review import summarize_roots


def sample(category, delta):
    return {
        "category": category,
        "fast_only": {"image_auroc": 0.7, "recall_at_fpr": 0.3, "fallback_rate": 0.0},
        "random_fallback": {"image_auroc": 0.7, "recall_at_fpr": 0.4, "fallback_rate": 0.5},
        "risk_fallback": {"image_auroc": 0.7 + delta, "recall_at_fpr": 0.4 + delta, "fallback_rate": 0.5},
        "full_only": {"image_auroc": 0.9, "recall_at_fpr": 0.8, "fallback_rate": 1.0},
        "routing": {"risk": {"actual_fallback_count": 2}, "random": {"actual_fallback_count": 2}},
    }


class SummarizeMvtecReviewTests(unittest.TestCase):
    def test_summary_reports_paired_delta_and_matching_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.json").write_text(json.dumps(sample("a", 0.1)), encoding="utf-8")
            (root / "b.json").write_text(json.dumps(sample("b", 0.2)), encoding="utf-8")
            result = summarize_roots([root], reps=200, seed=17)[str(root)]
            self.assertEqual(result["n_categories"], 2)
            self.assertEqual(result["paired_deltas"]["auroc_positive_categories"], 2)
            self.assertTrue(all(item["counts_match"] for item in result["routing_audit"]))


if __name__ == "__main__":
    unittest.main()
