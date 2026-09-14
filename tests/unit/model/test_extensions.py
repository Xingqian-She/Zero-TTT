from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from zero_ttt.config import load_config
from zero_ttt.model.depth_mixing import SparseDepthWeightedAverage
from zero_ttt.model.hypernet import scale_gradient


def test_sparse_depth_weighted_average_sources_and_identity() -> None:
    config = load_config("configs/test.toml")
    depth_config = replace(config.model.depth_mixing, dilation=4, period=4)
    mixing = SparseDepthWeightedAverage(8, depth_config)
    assert mixing.source_depths == {4: (0, 4), 8: (0, 4, 8)}
    states = {
        0: torch.full((1, 2, 3), 1.0),
        4: torch.full((1, 2, 3), 4.0, requires_grad=True),
        8: torch.full((1, 2, 3), 8.0, requires_grad=True),
    }
    output = mixing(8, states, states[8])
    assert torch.equal(output, states[8])
    output.sum().backward()
    assert mixing.weights["8"].grad is not None
    assert torch.count_nonzero(mixing.weights["8"].grad) == 3


@pytest.mark.parametrize("scale", (0.0, 0.1, 1.0))
def test_scale_gradient_is_forward_identity_with_scaled_backward(scale: float) -> None:
    source = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
    output = scale_gradient(source, scale)
    assert torch.equal(output, source)

    output.sum().backward()
    source_gradient = source.grad
    assert source_gradient is not None
    assert torch.equal(source_gradient, torch.full_like(source, scale))
