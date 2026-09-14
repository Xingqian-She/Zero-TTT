from __future__ import annotations

import torch
from torch import nn
from zero_ttt_trainer.ema import ema_decay, update_slow_weights


def test_sample_based_ema_uses_equivalent_batched_decay() -> None:
    fast = nn.Linear(2, 2, bias=False)
    slow = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        fast.weight.fill_(1.0)
        slow.weight.zero_()
    decay = update_slow_weights(slow, fast, samples=32, half_life_samples=32)
    assert decay == ema_decay(32, 32) == 0.5
    assert torch.allclose(slow.weight, torch.full_like(slow.weight, 0.5))


def test_ema_synchronizes_named_buffers() -> None:
    fast = nn.BatchNorm1d(2)
    slow = nn.BatchNorm1d(2)
    fast_running_mean = fast.running_mean
    fast_running_var = fast.running_var
    fast_batches_tracked = fast.num_batches_tracked
    slow_running_mean = slow.running_mean
    slow_running_var = slow.running_var
    slow_batches_tracked = slow.num_batches_tracked
    assert fast_running_mean is not None
    assert fast_running_var is not None
    assert fast_batches_tracked is not None
    assert slow_running_mean is not None
    assert slow_running_var is not None
    assert slow_batches_tracked is not None
    with torch.no_grad():
        fast_running_mean.copy_(torch.tensor([2.0, 3.0]))
        fast_running_var.copy_(torch.tensor([4.0, 5.0]))
        fast_batches_tracked.fill_(7)
        slow_running_mean.zero_()
        slow_running_var.fill_(1.0)
        slow_batches_tracked.zero_()
    update_slow_weights(slow, fast, samples=1, half_life_samples=1)
    assert torch.equal(slow_running_mean, fast_running_mean)
    assert torch.equal(slow_running_var, fast_running_var)
    assert torch.equal(slow_batches_tracked, fast_batches_tracked)


def test_fp32_ema_retains_small_updates() -> None:
    fast = nn.Linear(1, 1, bias=False)
    slow = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        slow.weight.fill_(1.0)
        fast.weight.fill_(1.001)
    update_slow_weights(slow, fast, samples=1, half_life_samples=1)
    assert slow.weight.dtype == torch.float32
    assert 1.0 < slow.weight.item() < fast.weight.item()
