"""Minimal reliability-constrained adaptive token allocation ViT."""

from dataclasses import dataclass
from math import ceil, log
from statistics import mean

import torch
from torch import nn
from sparse_attention import SparseTransformerBlock


@dataclass(frozen=True)
class RATAConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 192
    depth: int = 4
    heads: int = 3
    k_min: int = 32
    k_base: int = 98
    k_max: int = 196
    difficulty_gain: float = 0.5
    uncertainty_gain: float = 0.5
    adaptive: bool = True
    route_policy: str = "rata"


class RATABudget:
    def __init__(self, k_min, k_base, k_max, difficulty_gain=0.5, uncertainty_gain=0.5):
        if not 1 <= k_min <= k_base <= k_max:
            raise ValueError("require 1 <= k_min <= k_base <= k_max")
        self.k_min = int(k_min)
        self.k_base = int(k_base)
        self.k_max = int(k_max)
        self.difficulty_gain = float(difficulty_gain)
        self.uncertainty_gain = float(uncertainty_gain)

    @property
    def levels(self):
        return tuple(sorted(set((self.k_min, self.k_base, self.k_max))))

    def __call__(self, difficulty, uncertainty):
        raw = self.k_base * (
            1.0 + self.difficulty_gain * float(difficulty)
            + self.uncertainty_gain * float(uncertainty)
        )
        return max(self.k_min, min(self.k_max, int(round(raw))))

    def allocate(self, difficulty, uncertainty):
        """Allocate a separate integer budget for every sample in a batch."""
        difficulty = torch.as_tensor(difficulty, dtype=torch.float32)
        uncertainty = torch.as_tensor(uncertainty, dtype=torch.float32, device=difficulty.device)
        soft_budget = self.k_base * (
            1.0 + self.difficulty_gain * difficulty + self.uncertainty_gain * uncertainty
        )
        soft_budget = soft_budget.clamp(self.k_min, self.k_max)
        levels = torch.tensor(
            self.levels,
            dtype=soft_budget.dtype,
            device=soft_budget.device,
        )
        nearest = (soft_budget.unsqueeze(-1) - levels).abs().argmin(dim=-1)
        return levels[nearest].to(torch.long)


def risk_aware_loss(logits, labels, route, cheap_weight=0.25, risk_weight=0.5, budget_weight=0.01):
    """Train the cheap router to identify high-loss samples and preserve a budget target."""
    main_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    total = main_loss.mean()
    cheap_logits = route.get("cheap_logits")
    if cheap_logits is not None:
        cheap_loss = torch.nn.functional.cross_entropy(cheap_logits, labels, reduction="none")
        total = total + cheap_weight * cheap_loss.mean()
    difficulty = route.get("difficulty")
    if difficulty is not None:
        target = (main_loss.detach() / main_loss.detach().mean().clamp_min(1e-6)).clamp(0.0, 2.0) / 2.0
        total = total + risk_weight * torch.nn.functional.mse_loss(difficulty, target)
    soft_budget = route.get("soft_budget")
    if soft_budget is not None and cheap_logits is not None:
        total = total + budget_weight * soft_budget.float().mean()
    return total


def _normalized_entropy(probabilities):
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy / log(probabilities.shape[-1])


class RATAViT(nn.Module):
    def __init__(self, config: RATAConfig):
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.config = config
        self.grid = config.image_size // config.patch_size
        self.num_patches = self.grid * self.grid
        if not 1 <= config.k_min <= config.k_base <= config.k_max <= self.num_patches:
            raise ValueError("token budget exceeds the number of image patches")

        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.embed_dim))
        self.router = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.GELU(),
            nn.Linear(config.embed_dim // 2, 1),
        )
        self.cheap_head = nn.Linear(config.embed_dim, config.num_classes)
        self.encoder = nn.ModuleList(
            [SparseTransformerBlock(config.embed_dim, config.heads) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self.budget = RATABudget(
            config.k_min,
            config.k_base,
            config.k_max,
            config.difficulty_gain,
            config.uncertainty_gain,
        )

    def forward(self, images):
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        batch_size = images.shape[0]
        policy = self.config.route_policy
        if not self.config.adaptive and policy == "rata":
            policy = "full" if self.config.k_base == self.num_patches else "fixed_sparse"
        if policy in ("full", "fixed_sparse", "random_sparse"):
            difficulty = torch.zeros(images.shape[0], device=images.device)
            uncertainty = torch.zeros(images.shape[0], device=images.device)
            cheap_logits = None
            pooled_feature = patches.mean(dim=1)
        else:
            pooled_feature = patches.mean(dim=1)
            cheap_logits = self.cheap_head(pooled_feature)
            cheap_probabilities = cheap_logits.softmax(dim=-1)
            difficulty = 1.0 - cheap_probabilities.max(dim=-1).values
            uncertainty = _normalized_entropy(cheap_probabilities)
        soft_budget = self.config.k_base * (
            1.0 + self.config.difficulty_gain * difficulty
            + self.config.uncertainty_gain * uncertainty
        ).clamp(self.config.k_min / self.config.k_base, self.config.k_max / self.config.k_base)
        if policy == "full":
            token_counts = torch.full((batch_size,), self.num_patches, device=images.device, dtype=torch.long)
            selected_groups = [(torch.arange(batch_size, device=images.device), patches)]
        elif policy == "fixed_sparse":
            token_counts = torch.full((batch_size,), self.config.k_base, device=images.device, dtype=torch.long)
            selected_groups = [(torch.arange(batch_size, device=images.device), patches[:, : self.config.k_base])]
        elif policy == "random_sparse":
            k = self.config.k_base
            token_counts = torch.full((batch_size,), k, device=images.device, dtype=torch.long)
            random_indices = torch.rand(images.shape[0], self.num_patches, device=images.device).topk(k, dim=1).indices
            selected = patches.gather(1, random_indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))
            selected_groups = [(torch.arange(batch_size, device=images.device), selected)]
        elif policy in ("difficulty_only", "uncertainty_only", "rata"):
            token_counts = self.budget.allocate(difficulty, uncertainty)
            selected_groups = []
            token_scores = self.router(patches).squeeze(-1)
            if policy == "difficulty_only":
                token_scores = patches.norm(dim=-1) * difficulty.unsqueeze(1)
            elif policy == "uncertainty_only":
                token_scores = patches.norm(dim=-1) * uncertainty.unsqueeze(1)
            # Iterate over the small, fixed bucket set. Avoid GPU->CPU unique().tolist()
            # synchronization on every forward pass.
            for k_value in self.budget.levels:
                indices = (token_counts == k_value).nonzero(as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                top_indices = token_scores.index_select(0, indices).topk(k=k_value, dim=1).indices
                selected = patches.index_select(0, indices).gather(
                    1, top_indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
                )
                selected_groups.append((indices, selected))
        else:
            raise ValueError(f"unknown route policy: {policy}")
        logits = torch.empty(batch_size, self.config.num_classes, device=images.device, dtype=patches.dtype)
        for indices, selected in selected_groups:
            k = selected.shape[1]
            cls = self.cls_token.expand(selected.shape[0], -1, -1)
            sequence = torch.cat([cls, selected], dim=1)
            sequence = sequence + self.pos_embed[:, : k + 1]
            encoded = sequence
            for block in self.encoder:
                encoded = block(encoded)
            group_logits = self.head(self.norm(encoded[:, 0]))
            logits.index_copy_(0, indices, group_logits)
        route = {
            "token_count": int(token_counts.max().item()),
            "token_counts": token_counts,
            "token_keep_ratio": token_counts.float() / self.num_patches,
            "difficulty": difficulty.detach(),
            "uncertainty": uncertainty.detach(),
            "cheap_logits": cheap_logits,
            "soft_budget": soft_budget,
            "pooled_feature": pooled_feature,
        }
        return logits, route


def _percentile(values, q):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("latency values must not be empty")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_latencies(latencies_ms):
    values = list(latencies_ms)
    return {
        "count": len(values),
        "mean_ms": mean(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
    }
