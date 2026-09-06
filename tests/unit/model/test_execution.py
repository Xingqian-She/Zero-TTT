from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from zero_ttt.config import load_config
from zero_ttt.game.rules import BOARD_AREA
from zero_ttt.model.execution import BlockExecutor
from zero_ttt.model.interfaces import BlockOutput, NoOpBlockResidualPlugin
from zero_ttt.model.rope import AxialRoPE2D
from zero_ttt.model.tokens import TokenLayout


def test_checkpoint_executor_preserves_rng_during_recomputation() -> None:
    config = load_config("configs/test.toml")
    layout = TokenLayout(board_tokens=BOARD_AREA, special_tokens=1)
    rope = AxialRoPE2D(
        config.model.rope,
        config.model.d_model // config.model.n_heads,
        layout,
    )
    plugin = NoOpBlockResidualPlugin()

    class StochasticBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, hidden, rope, plugin):
            del rope, plugin
            output = F.dropout(hidden, p=0.5, training=True) * self.scale
            zero = output.new_zeros(())
            return BlockOutput(output, zero, zero, zero, zero)

    eager_block = StochasticBlock()
    checkpoint_block = StochasticBlock()
    checkpoint_block.load_state_dict(eager_block.state_dict())
    eager_executor = BlockExecutor(replace(config.execution, activation_checkpoint=False))
    checkpoint_executor = BlockExecutor(
        replace(
            config.execution,
            activation_checkpoint=True,
            activation_checkpoint_stride=1,
        )
    )
    eager_hidden = torch.randn(2, 3, requires_grad=True)
    checkpoint_hidden = eager_hidden.detach().clone().requires_grad_(True)
    torch.manual_seed(123)
    eager_output = eager_executor.run(
        index=0,
        training=True,
        block=eager_block,
        rope=rope,
        plugin=plugin,
        hidden=eager_hidden,
    )
    torch.manual_seed(123)
    checkpoint_output = checkpoint_executor.run(
        index=0,
        training=True,
        block=checkpoint_block,
        rope=rope,
        plugin=plugin,
        hidden=checkpoint_hidden,
    )
    eager_output.hidden.sum().backward()
    checkpoint_output.hidden.sum().backward()
    assert torch.equal(eager_output.hidden, checkpoint_output.hidden)
    assert torch.equal(eager_hidden.grad, checkpoint_hidden.grad)
    assert torch.equal(eager_block.scale.grad, checkpoint_block.scale.grad)
