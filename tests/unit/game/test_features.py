from __future__ import annotations

import numpy as np
from zero_ttt.config import load_config
from zero_ttt.game.features import encode_position
from zero_ttt.game.rules import BOARD_SIZE
from zero_ttt.game.state import GameState


def point(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def test_feature_schema_and_current_player_view() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game).play(point(3, 3))
    features = encode_position(state)
    assert features.board.shape == (25, 19, 19)
    assert features.global_features.shape == (5,)
    assert features.legal.shape == (362,)
    assert features.board.dtype == np.float32
    assert features.global_features.dtype == np.float32
    assert features.legal.dtype == np.bool_
    assert features.board[1, 3, 3] == 1.0  # black is the opponent from White's view
    assert features.global_features[4] == 1.0
