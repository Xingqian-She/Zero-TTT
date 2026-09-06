from __future__ import annotations

import pytest
from zero_ttt.config import load_config
from zero_ttt.game.rules import BOARD_SIZE, PASS_ACTION, Color
from zero_ttt.game.state import GameState


def point(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def test_two_passes_end_game_and_komi_scores() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game).play(PASS_ACTION).play(PASS_ACTION)
    assert state.is_terminal()
    assert state.termination_reason() == "two_passes"
    result = state.score()
    assert result.score.margin_half_points == -15
    assert result.winner is Color.WHITE


def test_move_limit_forces_terminal_state() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game)
    for _ in range(config.game.max_moves):
        if state.is_terminal():
            break
        legal = state.legal_actions()
        action = next(index for index, allowed in enumerate(legal[:-1]) if allowed)
        state = state.play(action)
    assert state.is_terminal()
    assert state.termination_reason() == "move_limit"


def test_illegal_action_raises() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game).play(point(0, 0))
    with pytest.raises(ValueError, match="illegal action"):
        state.play(point(0, 0))
