import csv
import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1]))

import make_strong_routing_figures as figure_module
from make_strong_routing_figures import generate_figures


class StrongRoutingFigureTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _write_main_table(self):
        fields = [
            "dataset", "unit_count", "random_auroc_mean", "fast_score_auroc_mean",
            "uncertainty_auroc_mean", "risk_auroc_mean", "risk_random_delta_mean",
            "risk_random_delta_low", "risk_random_delta_high", "risk_fast_score_delta_mean",
            "risk_fast_score_delta_low", "risk_fast_score_delta_high",
            "risk_uncertainty_delta_mean", "risk_uncertainty_delta_low",
            "risk_uncertainty_delta_high",
        ]
        rows = []
        for dataset, count in (("MVTec", 45), ("MPDD", 18), ("VisA", 36)):
            rows.append({
                "dataset": dataset, "unit_count": count,
                "random_auroc_mean": 0.7, "fast_score_auroc_mean": 0.8,
                "uncertainty_auroc_mean": 0.75, "risk_auroc_mean": 0.82,
                "risk_random_delta_mean": 0.01, "risk_random_delta_low": 0.0,
                "risk_random_delta_high": 0.02, "risk_fast_score_delta_mean": 0.01,
                "risk_fast_score_delta_low": 0.0, "risk_fast_score_delta_high": 0.02,
                "risk_uncertainty_delta_mean": 0.01, "risk_uncertainty_delta_low": 0.0,
                "risk_uncertainty_delta_high": 0.02,
            })
        path = self.root / "main_results.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_generate_publication_outputs(self):
        self._write_main_table()
        outputs = generate_figures(self.root, self.root / "figures")
        base = outputs["strong_routing"]
        self.assertEqual(
            {path.suffix for path in self.root.joinpath("figures").glob("fig_canonical_v3_strong_routing.*")},
            {".pdf", ".png", ".svg", ".tiff"},
        )
        self.assertTrue(base.with_suffix(".pdf").stat().st_size > 0)

    def test_reject_missing_dataset(self):
        path = self._write_main_table()
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]), encoding="utf-8")
        with self.assertRaises(ValueError):
            generate_figures(self.root, self.root / "figures")

    def test_legends_stay_outside_plotting_areas(self):
        self._write_main_table()
        captured = {}

        def capture_figure(fig, _base_path):
            captured["fig"] = fig

        original = figure_module._save_publication_figure
        figure_module._save_publication_figure = capture_figure
        try:
            generate_figures(self.root, self.root / "figures")
        finally:
            figure_module._save_publication_figure = original

        fig = captured["fig"]
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for panel in fig.axes:
                legend_box = panel.get_legend().get_window_extent(renderer)
                plotting_box = panel.get_window_extent(renderer)
                self.assertFalse(legend_box.overlaps(plotting_box))
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
