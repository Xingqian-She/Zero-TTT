from __future__ import annotations

import numpy as np
import pytest
import torch
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE
from zero_ttt_dataset import SyntheticBatchSource, TrainBatch
from zero_ttt_selfplay_worker.inference.contracts import (
    InferenceBatch,
    InferenceOutput,
    PositionEvaluator,
)


class FakeEvaluator:
    model_version = "test-v1"

    def evaluate(self, batch: InferenceBatch) -> InferenceOutput:
        count = batch.board.shape[0]
        return InferenceOutput(
            policy_logits=torch.zeros(count, ACTION_SIZE),
            value=torch.zeros(count),
            ownership=torch.zeros(count, BOARD_AREA),
            score_margin=torch.zeros(count),
        )


def test_synthetic_source_implements_training_boundary() -> None:
    batch = SyntheticBatchSource().next_batch(3, np.random.default_rng(4))
    assert isinstance(batch, TrainBatch)
    assert batch.board.shape == (3, 25, BOARD_SIZE, BOARD_SIZE)
    assert all(
        array.dtype == np.float32
        for array in (
            batch.board,
            batch.global_features,
            batch.policy,
            batch.value,
            batch.ownership,
            batch.score_margin,
        )
    )
    assert all(
        array.dtype == np.bool_
        for array in (batch.legal, batch.value_mask, batch.ownership_mask, batch.score_mask)
    )
    assert np.allclose(batch.policy.sum(axis=1), 1.0)


def test_train_batch_rejects_policy_on_illegal_action() -> None:
    source = SyntheticBatchSource()
    batch = source.next_batch(1, np.random.default_rng(1))
    illegal = batch.policy.argmax(axis=1)
    legal = batch.legal.copy()
    legal[0, illegal[0]] = False
    with pytest.raises(ValueError, match="illegal"):
        TrainBatch(
            board=batch.board,
            global_features=batch.global_features,
            legal=legal,
            policy=batch.policy,
            value=batch.value,
            ownership=batch.ownership,
            score_margin=batch.score_margin,
            value_mask=batch.value_mask,
            ownership_mask=batch.ownership_mask,
            score_mask=batch.score_mask,
        )


def test_position_evaluator_is_structural_and_batched() -> None:
    evaluator = FakeEvaluator()
    assert isinstance(evaluator, PositionEvaluator)
    batch = InferenceBatch(
        board=torch.zeros(2, 25, BOARD_SIZE, BOARD_SIZE),
        global_features=torch.zeros(2, 5),
        legal=torch.ones(2, ACTION_SIZE, dtype=torch.bool),
    )
    output = evaluator.evaluate(batch)
    assert output.policy_logits.shape == (2, ACTION_SIZE)
    assert output.ownership is not None
