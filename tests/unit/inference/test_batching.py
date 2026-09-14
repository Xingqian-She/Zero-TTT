from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from zero_ttt.config import load_config
from zero_ttt.game.state import GameState
from zero_ttt_selfplay_worker.inference import (
    BatchedInferenceBroker,
    InferenceBatch,
    InferenceOutput,
)


class CountingEvaluator:
    model_version = "counting-v1"

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.lock = threading.Lock()

    def evaluate(self, batch: InferenceBatch) -> InferenceOutput:
        size = batch.board.shape[0]
        with self.lock:
            self.calls.append(size)
        return InferenceOutput(
            policy_logits=torch.zeros(size, 362),
            value=torch.zeros(size),
            ownership=torch.zeros(size, 361),
            score_margin=torch.zeros(size),
        )


class FailingEvaluator:
    model_version = "failing-v1"

    def evaluate(self, batch: InferenceBatch) -> InferenceOutput:
        del batch
        raise RuntimeError("inference failed")


def test_broker_forms_one_full_batch_and_caches() -> None:
    config = load_config("configs/test.toml")
    states = [GameState.new(config.game).play(action) for action in range(16)]
    evaluator = CountingEvaluator()
    with BatchedInferenceBroker(
        evaluator, batch_size=16, batch_wait_ms=50, cache_size=32
    ) as broker:
        with ThreadPoolExecutor(max_workers=16) as pool:
            values = list(pool.map(broker.evaluate, states))
        assert len(values) == 16
        assert evaluator.calls == [16]
        broker.evaluate(states[0])
        assert evaluator.calls == [16]
        assert broker.stats.full_batches == 1
        assert broker.stats.cache_hits >= 1


def test_broker_flushes_partial_batch_without_deadlock() -> None:
    config = load_config("configs/test.toml")
    evaluator = CountingEvaluator()
    with BatchedInferenceBroker(evaluator, batch_size=16, batch_wait_ms=0, cache_size=4) as broker:
        result = broker.evaluate(GameState.new(config.game))
        assert result.policy_logits.shape == (362,)
        assert result.policy_logits.dtype == np.float32
        assert result.ownership is not None
        assert result.ownership.dtype == np.float32
        assert evaluator.calls == [1]


def test_broker_deduplicates_same_state_and_propagates_errors() -> None:
    config = load_config("configs/test.toml")
    state = GameState.new(config.game)
    evaluator = CountingEvaluator()
    with BatchedInferenceBroker(evaluator, batch_size=16, batch_wait_ms=50, cache_size=4) as broker:
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(broker.evaluate, [state] * 16))
        assert len(results) == 16
        assert evaluator.calls == [1]

    broker = BatchedInferenceBroker(
        FailingEvaluator(), batch_size=16, batch_wait_ms=50, cache_size=4
    )
    states = [state.play(action) for action in range(16)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(broker.evaluate, item) for item in states]
        for future in futures:
            with pytest.raises(RuntimeError, match="inference failed"):
                future.result(timeout=2)
    broker.close()
    broker.close()
    with pytest.raises(RuntimeError, match="closed"):
        broker.evaluate(state)


def test_inference_contracts_reject_non_fp32_tensors() -> None:
    with pytest.raises(TypeError, match="float32"):
        InferenceBatch(
            board=torch.zeros(1, 25, 19, 19, dtype=torch.float64),
            global_features=torch.zeros(1, 5, dtype=torch.float64),
            legal=torch.ones(1, 362, dtype=torch.bool),
        )
    with pytest.raises(TypeError, match="float32"):
        InferenceOutput(
            policy_logits=torch.zeros(1, 362, dtype=torch.float64),
            value=torch.zeros(1, dtype=torch.float64),
        )
