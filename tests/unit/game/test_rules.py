from __future__ import annotations

import numpy as np
from zero_ttt.game.rules import (
    BOARD_AREA,
    BOARD_SIZE,
    PASS_ACTION,
    Color,
    area_score,
    legal_actions,
    play_point,
)


def point(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def board_with(stones: dict[int, Color]) -> bytes:
    board = bytearray(BOARD_AREA)
    for location, color in stones.items():
        board[location] = int(color)
    return bytes(board)


def test_capture_and_single_stone_suicide() -> None:
    center = point(1, 1)
    capture_at = point(2, 1)
    board = board_with(
        {
            center: Color.WHITE,
            point(1, 0): Color.BLACK,
            point(0, 1): Color.BLACK,
            point(1, 2): Color.BLACK,
        }
    )
    result = play_point(board, Color.BLACK, capture_at, {board})
    assert result is not None
    assert result[center] == 0
    assert result[capture_at] == int(Color.BLACK)

    suicide_board = board_with(
        {
            point(1, 0): Color.BLACK,
            point(0, 1): Color.BLACK,
            point(1, 2): Color.BLACK,
            point(2, 1): Color.BLACK,
        }
    )
    assert play_point(suicide_board, Color.WHITE, center, {suicide_board}) is None


def test_tromp_taylor_allows_multi_stone_suicide() -> None:
    old_stone = point(1, 1)
    action = point(2, 1)
    board = board_with(
        {
            old_stone: Color.WHITE,
            point(0, 1): Color.BLACK,
            point(1, 0): Color.BLACK,
            point(1, 2): Color.BLACK,
            point(2, 0): Color.BLACK,
            point(2, 2): Color.BLACK,
            point(3, 1): Color.BLACK,
        }
    )
    result = play_point(board, Color.WHITE, action, {board})
    assert result is not None
    assert result[old_stone] == 0
    assert result[action] == 0
    assert legal_actions(board, Color.WHITE, {board})[action]


def test_positional_superko_can_forbid_multi_stone_suicide_result() -> None:
    old_stone = point(1, 1)
    action = point(2, 1)
    board = board_with(
        {
            old_stone: Color.WHITE,
            point(0, 1): Color.BLACK,
            point(1, 0): Color.BLACK,
            point(1, 2): Color.BLACK,
            point(2, 0): Color.BLACK,
            point(2, 2): Color.BLACK,
            point(3, 1): Color.BLACK,
        }
    )
    result = play_point(board, Color.WHITE, action, {board})
    assert result is not None
    assert play_point(board, Color.WHITE, action, {board, result}) is None


def test_one_move_can_capture_multiple_groups() -> None:
    action = point(1, 1)
    left = point(1, 0)
    above = point(0, 1)
    board = board_with(
        {
            left: Color.WHITE,
            above: Color.WHITE,
            point(0, 0): Color.BLACK,
            point(2, 0): Color.BLACK,
            point(0, 2): Color.BLACK,
        }
    )
    result = play_point(board, Color.BLACK, action, {board})
    assert result is not None
    assert result[left] == 0
    assert result[above] == 0


def test_positional_superko_forbids_immediate_ko_repetition() -> None:
    before = board_with(
        {
            point(0, 1): Color.WHITE,
            point(1, 0): Color.WHITE,
            point(0, 2): Color.BLACK,
            point(1, 1): Color.BLACK,
        }
    )
    after = play_point(before, Color.BLACK, point(0, 0), {before})
    assert after is not None
    assert play_point(after, Color.WHITE, point(0, 1), {after}) == before
    assert play_point(after, Color.WHITE, point(0, 1), {before, after}) is None


def test_positional_superko_uses_exact_board_and_pass_is_always_legal() -> None:
    empty = bytes(BOARD_AREA)
    once = play_point(empty, Color.BLACK, point(3, 3), {empty})
    assert once is not None
    assert play_point(empty, Color.BLACK, point(3, 3), {empty, once}) is None
    mask = legal_actions(empty, Color.BLACK, {empty, once})
    assert not mask[point(3, 3)]
    assert mask[PASS_ACTION]


def test_area_score_ownership_is_signed() -> None:
    board = board_with({point(0, 0): Color.BLACK, point(18, 18): Color.WHITE})
    score = area_score(board, 15)
    ownership = np.frombuffer(score.ownership, dtype=np.int8)
    assert ownership[point(0, 0)] == 1
    assert ownership[point(18, 18)] == -1
    assert ownership[point(9, 9)] == 0
