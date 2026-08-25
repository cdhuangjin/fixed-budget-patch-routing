import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from evaluate_fixed_protocol import parse_checkpoint_path


class FixedProtocolTests(unittest.TestCase):
    def test_checkpoint_path_parser_extracts_seed_and_method(self):
        path = Path("/runs/seed_17/rata/checkpoint.pt")
        parsed = parse_checkpoint_path(path)
        self.assertEqual(parsed, {"seed": 17, "method": "rata"})


if __name__ == "__main__":
    unittest.main()
