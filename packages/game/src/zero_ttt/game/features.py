"""Shared model feature encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zero_ttt.game.rules import BOARD_AREA, BOARD_SIZE, Color, analyze_board
from zero_ttt.game.state import GameState

POINT_FEATURES = 25
GLOBAL_FEATURES = 5
FEATURE_SCHEMA_ID = "zero-ttt-position-features-25x19x19-global5-v1"


@dataclass(frozen=True, slots=True)
class PositionFeatures:
    board: NDArray[np.float32]
    global_features: NDArray[np.float32]
    legal: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.board.shape != (POINT_FEATURES, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"board features must have shape {(POINT_FEATURES, BOARD_SIZE, BOARD_SIZE)}"
            )
        if self.global_features.shape != (GLOBAL_FEATURES,):
            raise ValueError(f"global features must have shape {(GLOBAL_FEATURES,)}")
        if self.legal.shape != (BOARD_AREA + 1,):
            raise ValueError(f"legal mask must have shape {(BOARD_AREA + 1,)}")


def encode_position(state: GameState) -> PositionFeatures:
    planes = np.zeros((POINT_FEATURES, BOARD_AREA), dtype=np.float32)
    own = int(state.to_play)
    opponent = int(state.to_play.opponent)
    empty = bytes(BOARD_AREA)
    histories = (*state.recent_boards, *(empty for _ in range(8 - len(state.recent_boards))))
    for index, board in enumerate(histories[:8]):
        values = np.frombuffer(board, dtype=np.uint8)
        planes[2 * index] = values == own
        planes[2 * index + 1] = values == opponent

    analysis = analyze_board(state.board)
    for point, value in enumerate(state.board):
        if not value:
            continue
        group_id = analysis.group_at[point]
        liberty_bucket = min(len(analysis.group_liberties[group_id]), 3) - 1
        if value == own:
            planes[16 + liberty_bucket, point] = 1.0
        else:
            planes[19 + liberty_bucket, point] = 1.0

    for offset, action in enumerate(state.recent_moves[:2]):
        if action < BOARD_AREA:
            planes[22 + offset, action] = 1.0

    legal = np.asarray(state.legal_actions(), dtype=np.bool_)
    planes[24] = legal[:BOARD_AREA]
    komi_points = state.komi_half_points / 2.0
    signed_komi = komi_points if state.to_play is Color.WHITE else -komi_points
    global_features = np.asarray(
        [
            signed_komi / 20.0,
            state.move_number / 722.0,
            state.consecutive_passes / 2.0,
            float(state.to_play is Color.BLACK),
            float(state.to_play is Color.WHITE),
        ],
        dtype=np.float32,
    )
    return PositionFeatures(
        board=planes.reshape(POINT_FEATURES, BOARD_SIZE, BOARD_SIZE),
        global_features=global_features,
        legal=legal,
    )
