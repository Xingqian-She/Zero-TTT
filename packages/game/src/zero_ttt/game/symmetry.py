"""The eight D4 symmetries of a square Go board."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zero_ttt.game.features import PositionFeatures
from zero_ttt.game.rules import BOARD_AREA, BOARD_SIZE, PASS_ACTION

SYMMETRY_COUNT = 8


def transform_point(point: int, symmetry: int) -> int:
    if not 0 <= symmetry < SYMMETRY_COUNT:
        raise ValueError("symmetry must be in [0, 8)")
    if point == PASS_ACTION:
        return PASS_ACTION
    if not 0 <= point < BOARD_AREA:
        raise ValueError("point is outside the board")
    row, col = divmod(point, BOARD_SIZE)
    if symmetry >= 4:
        col = BOARD_SIZE - 1 - col
    for _ in range(symmetry % 4):
        row, col = col, BOARD_SIZE - 1 - row
    return row * BOARD_SIZE + col


def transform_spatial[ScalarT: np.generic](
    array: NDArray[ScalarT], symmetry: int
) -> NDArray[ScalarT]:
    if not 0 <= symmetry < SYMMETRY_COUNT:
        raise ValueError("symmetry must be in [0, 8)")
    result = array
    if symmetry >= 4:
        result = np.flip(result, axis=-1)
    if symmetry % 4:
        result = np.rot90(result, k=-(symmetry % 4), axes=(-2, -1))
    return np.ascontiguousarray(result)


def transform_action_vector[ScalarT: np.generic](
    vector: NDArray[ScalarT], symmetry: int
) -> NDArray[ScalarT]:
    if vector.shape[-1] != BOARD_AREA + 1:
        raise ValueError("action vector must end in 362 entries")
    result = np.empty_like(vector)
    for action in range(BOARD_AREA):
        result[..., transform_point(action, symmetry)] = vector[..., action]
    result[..., PASS_ACTION] = vector[..., PASS_ACTION]
    return result


@dataclass(frozen=True, slots=True)
class AugmentedSample:
    features: PositionFeatures
    policy: NDArray[np.float32]
    ownership: NDArray[np.float32]


def augment_sample(
    features: PositionFeatures,
    policy: NDArray[np.float32],
    ownership: NDArray[np.float32],
    symmetry: int,
) -> AugmentedSample:
    board = transform_spatial(features.board, symmetry)
    legal = transform_action_vector(features.legal, symmetry)
    transformed_policy = transform_action_vector(policy, symmetry)
    ownership_board = ownership.reshape(BOARD_SIZE, BOARD_SIZE)
    transformed_ownership = transform_spatial(ownership_board, symmetry).reshape(BOARD_AREA)
    return AugmentedSample(
        features=PositionFeatures(
            board=board,
            global_features=features.global_features.copy(),
            legal=legal,
        ),
        policy=transformed_policy,
        ownership=transformed_ownership,
    )
