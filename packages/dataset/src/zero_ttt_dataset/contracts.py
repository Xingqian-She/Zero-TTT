"""Stable boundary between training workflows and the generic trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from zero_ttt.game.features import GLOBAL_FEATURES, POINT_FEATURES
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE


@dataclass(frozen=True, slots=True)
class TrainBatch:
    """One optimizer microbatch in the current-player perspective.

    Ownership and score labels are optional per sample. Their masks are false
    when a source cannot provide the corresponding target.
    """

    board: NDArray[np.float32]
    global_features: NDArray[np.float32]
    legal: NDArray[np.bool_]
    policy: NDArray[np.float32]
    value: NDArray[np.float32]
    ownership: NDArray[np.float32]
    score_margin: NDArray[np.float32]
    value_mask: NDArray[np.bool_]
    ownership_mask: NDArray[np.bool_]
    score_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.board.ndim != 4:
            raise ValueError("board must be a batched rank-4 array")
        batch = self.board.shape[0]
        expected = {
            "board": ((batch, POINT_FEATURES, BOARD_SIZE, BOARD_SIZE), np.float32),
            "global_features": ((batch, GLOBAL_FEATURES), np.float32),
            "legal": ((batch, ACTION_SIZE), np.bool_),
            "policy": ((batch, ACTION_SIZE), np.float32),
            "value": ((batch,), np.float32),
            "ownership": ((batch, BOARD_AREA), np.float32),
            "score_margin": ((batch,), np.float32),
            "value_mask": ((batch,), np.bool_),
            "ownership_mask": ((batch,), np.bool_),
            "score_mask": ((batch,), np.bool_),
        }
        if batch <= 0:
            raise ValueError("a training batch cannot be empty")
        for name, (shape, dtype) in expected.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{name} must have shape {shape} and dtype {dtype}")
        numeric = (
            self.board,
            self.global_features,
            self.policy,
            self.value,
            self.ownership,
            self.score_margin,
        )
        if not all(np.isfinite(value).all() for value in numeric):
            raise ValueError("core training targets must be finite")
        if np.any(self.policy < 0) or not np.allclose(self.policy.sum(axis=1), 1.0):
            raise ValueError("policy rows must be normalized non-negative distributions")
        if np.any(self.policy[~self.legal] != 0):
            raise ValueError("policy mass cannot be assigned to illegal actions")
        if not np.all(self.legal.any(axis=1)):
            raise ValueError("every position needs at least one legal action")


class BatchSource(Protocol):
    """A workflow-specific provider of normalized training batches."""

    def next_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> TrainBatch: ...
