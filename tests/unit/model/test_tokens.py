from __future__ import annotations

import pytest
import torch
from zero_ttt.model.tokens import TokenLayout


def test_token_layout_selects_board_special_and_summary_tokens() -> None:
    layout = TokenLayout(board_tokens=4, special_tokens=3, summary_offset=1)
    values = torch.arange(14, dtype=torch.float32).reshape(2, 7, 1)

    assert layout.board_slice == slice(0, 4)
    assert layout.special_slice == slice(4, 7)
    assert layout.summary_index == 5
    assert layout.total_tokens == 7
    assert torch.equal(layout.board(values), values[:, :4])
    assert torch.equal(layout.special(values), values[:, 4:])
    assert torch.equal(layout.summary(values), values[:, 5])


@pytest.mark.parametrize(
    "kwargs",
    (
        {"board_tokens": 0, "special_tokens": 1},
        {"board_tokens": 4, "special_tokens": 0},
        {"board_tokens": 4, "special_tokens": 2, "summary_offset": -1},
        {"board_tokens": 4, "special_tokens": 2, "summary_offset": 2},
    ),
)
def test_token_layout_rejects_invalid_definitions(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        TokenLayout(**kwargs)


@pytest.mark.parametrize("shape", ((7,), (2, 6, 4), (2, 8, 4)))
def test_token_layout_rejects_invalid_tensor_shapes(shape: tuple[int, ...]) -> None:
    layout = TokenLayout(board_tokens=4, special_tokens=3)
    with pytest.raises(ValueError, match="expected 7 tokens"):
        layout.validate(torch.zeros(shape))
