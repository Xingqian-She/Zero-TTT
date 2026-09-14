from __future__ import annotations

import numpy as np
from zero_ttt.config import load_config
from zero_ttt.game.features import encode_position
from zero_ttt.game.rules import BOARD_SIZE, PASS_ACTION
from zero_ttt.game.state import GameState
from zero_ttt.game.symmetry import (
    augment_sample,
    transform_action_vector,
    transform_point,
)


def point(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def test_d4_action_and_sample_transforms_are_consistent() -> None:
    config = load_config("configs/test.toml")
    features = encode_position(GameState.new(config.game))
    policy = np.arange(362, dtype=np.float32)
    ownership = np.arange(361, dtype=np.float32)
    for symmetry in range(8):
        transformed = augment_sample(features, policy, ownership, symmetry)
        expected_action = transform_point(point(2, 5), symmetry)
        assert transformed.policy[expected_action] == policy[point(2, 5)]
        assert transformed.ownership[expected_action] == ownership[point(2, 5)]
        assert transformed.policy[PASS_ACTION] == policy[PASS_ACTION]
        assert transformed.features.board.dtype == features.board.dtype
        assert transformed.features.legal.dtype == features.legal.dtype
        assert transformed.policy.dtype == policy.dtype
        assert transformed.ownership.dtype == ownership.dtype
        assert np.array_equal(
            transformed.features.legal,
            transform_action_vector(features.legal, symmetry),
        )
