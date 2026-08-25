import numpy as np
import pytest
import torch

pytest.importorskip("torchvision")
from evaluate_mvtec_patchcore import image_coreset_bank


def test_image_coreset_is_deterministic_and_keeps_patch_shape():
    features = torch.arange(12 * 3 * 4, dtype=torch.float32).reshape(12, 3, 4)
    bank_a, indices_a = image_coreset_bank(features, ratio=0.25, seed=17)
    bank_b, indices_b = image_coreset_bank(features, ratio=0.25, seed=17)
    assert bank_a.shape == (len(indices_a) * features.shape[1], features.shape[-1])
    assert len(indices_a) == 3
    assert len(np.unique(indices_a)) == 3
    assert np.array_equal(indices_a, indices_b)
    assert torch.equal(bank_a, bank_b)
