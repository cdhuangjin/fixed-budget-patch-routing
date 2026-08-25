import sys
from pathlib import Path

import math
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from rata_vit import RATABudget, RATAConfig, RATAViT, risk_aware_loss, summarize_latencies
from sparse_attention import SparseSelfAttention, SparseTransformerBlock


class RATAUnitTests(unittest.TestCase):
    def test_sparse_self_attention_preserves_shape_and_gradients(self):
        attention = SparseSelfAttention(embed_dim=32, heads=4)
        tokens = torch.randn(2, 9, 32, requires_grad=True)
        output = attention(tokens)
        self.assertEqual(tuple(output.shape), (2, 9, 32))
        output.square().mean().backward()
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_sparse_transformer_block_accepts_different_token_counts(self):
        block = SparseTransformerBlock(embed_dim=32, heads=4)
        short = block(torch.randn(2, 5, 32))
        long = block(torch.randn(2, 17, 32))
        self.assertEqual(tuple(short.shape), (2, 5, 32))
        self.assertEqual(tuple(long.shape), (2, 17, 32))

    def test_budget_is_bounded_and_monotone_in_risk(self):
        budget = RATABudget(k_min=4, k_base=8, k_max=16, difficulty_gain=1.0, uncertainty_gain=1.0)

        easy = budget(difficulty=0.0, uncertainty=0.0)
        hard = budget(difficulty=1.0, uncertainty=1.0)

        self.assertEqual(easy, 8)
        self.assertTrue(4 <= easy <= 16)
        self.assertTrue(4 <= hard <= 16)
        self.assertGreaterEqual(hard, easy)

    def test_budget_allocates_per_sample_not_batch_mean(self):
        budget = RATABudget(k_min=4, k_base=8, k_max=16, difficulty_gain=1.0, uncertainty_gain=0.0)
        allocations = budget.allocate(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]))
        self.assertEqual(allocations.tolist(), [8, 16])

    def test_rata_vit_returns_logits_and_route_metadata(self):
        model = RATAViT(
            RATAConfig(
                image_size=32,
                patch_size=4,
                in_channels=3,
                num_classes=10,
                embed_dim=32,
                depth=2,
                heads=4,
                k_min=4,
                k_base=16,
                k_max=32,
            )
        )

        logits, route = model(torch.randn(2, 3, 32, 32))

        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(4 <= route["token_count"] <= 64)
        self.assertEqual(tuple(route["difficulty"].shape), (2,))
        self.assertEqual(tuple(route["uncertainty"].shape), (2,))

    def test_adaptive_route_exposes_samplewise_budget_and_cheap_logits(self):
        config = RATAConfig(
            image_size=32, patch_size=4, num_classes=10, embed_dim=32,
            depth=1, heads=4, k_min=4, k_base=16, k_max=64,
            difficulty_gain=2.0, uncertainty_gain=0.0,
        )
        model = RATAViT(config)
        _, route = model(torch.randn(3, 3, 32, 32))
        self.assertEqual(tuple(route["token_counts"].shape), (3,))
        self.assertEqual(tuple(route["soft_budget"].shape), (3,))
        self.assertEqual(tuple(route["cheap_logits"].shape), (3, 10))

    def test_all_first_stage_paths_expose_pooled_feature(self):
        for policy, adaptive in (("full", False), ("fixed_sparse", False), ("rata", True)):
            config = RATAConfig(
                image_size=32, patch_size=4, num_classes=10, embed_dim=32,
                depth=1, heads=4, k_min=4, k_base=16, k_max=32,
                adaptive=adaptive, route_policy=policy,
            )
            model = RATAViT(config)
            _, route = model(torch.randn(2, 3, 32, 32))
            self.assertEqual(tuple(route["pooled_feature"].shape), (2, 32))
            self.assertIn("cheap_logits", route)
            self.assertIn("uncertainty", route)

    def test_risk_aware_loss_backpropagates_through_cheap_router(self):
        cheap_logits = torch.randn(2, 10, requires_grad=True)
        logits = torch.randn(2, 10, requires_grad=True)
        labels = torch.tensor([1, 3])
        route = {
            "cheap_logits": cheap_logits,
            "difficulty": torch.tensor([0.2, 0.8]),
            "soft_budget": torch.tensor([8.0, 14.0], requires_grad=True),
        }
        loss = risk_aware_loss(logits, labels, route)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(cheap_logits.grad)

    def test_latency_summary_reports_percentiles(self):
        summary = summarize_latencies([1.0, 2.0, 3.0, 4.0, 5.0])

        self.assertEqual(summary["count"], 5)
        self.assertTrue(math.isclose(summary["p50_ms"], 3.0))
        self.assertTrue(math.isclose(summary["p95_ms"], 4.8))

    def test_nonadaptive_mode_bypasses_router(self):
        config = RATAConfig(
            image_size=32,
            patch_size=4,
            embed_dim=32,
            depth=1,
            heads=4,
            num_classes=10,
            k_min=32,
            k_base=32,
            k_max=32,
            adaptive=False,
        )
        model = RATAViT(config)
        calls = {"cheap": 0, "router": 0}
        model.cheap_head.register_forward_hook(lambda *_: calls.__setitem__("cheap", calls["cheap"] + 1))
        model.router.register_forward_hook(lambda *_: calls.__setitem__("router", calls["router"] + 1))
        model(torch.randn(2, 3, 32, 32))
        self.assertEqual(calls, {"cheap": 0, "router": 0})


if __name__ == "__main__":
    unittest.main()
