"""Thread-safe batching of blocking game-state evaluation requests."""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from zero_ttt.game.features import encode_position
from zero_ttt.game.state import GameState

from zero_ttt_selfplay_worker.inference.contracts import InferenceBatch, PositionEvaluator


@dataclass(frozen=True, slots=True)
class StateEvaluation:
    policy_logits: NDArray[np.float32]
    value: float
    ownership: NDArray[np.float32] | None
    score_margin: float | None


@dataclass(frozen=True, slots=True)
class BatchingStats:
    requests: int
    cache_hits: int
    batches: int
    real_evaluations: int
    padded_evaluations: int
    full_batches: int
    mean_batch_latency_ms: float
    max_batch_latency_ms: float
    real_batch_fraction: float
    full_batch_fraction: float


@dataclass(slots=True)
class _Request:
    state: GameState
    future: Future[StateEvaluation]


class BatchedInferenceBroker:
    """Expose one blocking state call while one thread owns model execution."""

    def __init__(
        self,
        evaluator: PositionEvaluator,
        *,
        batch_size: int = 16,
        batch_wait_ms: float = 2.0,
        cache_size: int = 32768,
    ) -> None:
        if batch_size <= 0 or batch_wait_ms < 0 or cache_size <= 0:
            raise ValueError("invalid batching configuration")
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.batch_wait_seconds = batch_wait_ms / 1000.0
        self.cache_size = cache_size
        self._queue: queue.Queue[_Request | None] = queue.Queue()
        self._cache: OrderedDict[tuple[object, ...], StateEvaluation] = OrderedDict()
        self._lock = threading.Lock()
        self._closed = False
        self._requests = 0
        self._cache_hits = 0
        self._batches = 0
        self._real_evaluations = 0
        self._full_batches = 0
        self._inference_seconds = 0.0
        self._max_inference_seconds = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name="zero-ttt-inference",
            daemon=True,
        )
        self._thread.start()

    def _cache_get(self, identity: tuple[object, ...]) -> StateEvaluation | None:
        with self._lock:
            value = self._cache.get(identity)
            if value is not None:
                self._cache.move_to_end(identity)
                self._cache_hits += 1
            return value

    def _cache_put(self, identity: tuple[object, ...], value: StateEvaluation) -> None:
        with self._lock:
            self._cache[identity] = value
            self._cache.move_to_end(identity)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def evaluate(self, state: GameState) -> StateEvaluation:
        identity = state.identity()
        cached = self._cache_get(identity)
        if cached is not None:
            return cached
        future: Future[StateEvaluation] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("inference broker is closed")
            self._requests += 1
            self._queue.put(_Request(state, future))
        return future.result()

    def _run(self) -> None:
        stopping = False
        while not stopping:
            first = self._queue.get()
            if first is None:
                return
            requests, stopping = self._collect_requests(first)
            groups = self._uncached_groups(requests)
            if not groups:
                continue
            try:
                self._evaluate_groups(groups)
            except Exception as error:
                self._fail_groups(groups, error)

    def _collect_requests(self, first: _Request) -> tuple[list[_Request], bool]:
        requests = [first]
        deadline = time.monotonic() + self.batch_wait_seconds
        while len(requests) < self.batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                request = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if request is None:
                return requests, True
            requests.append(request)
        return requests, False

    def _uncached_groups(
        self, requests: list[_Request]
    ) -> OrderedDict[tuple[object, ...], list[_Request]]:
        groups: OrderedDict[tuple[object, ...], list[_Request]] = OrderedDict()
        for request in requests:
            identity = request.state.identity()
            cached = self._cache_get(identity)
            if cached is None:
                groups.setdefault(identity, []).append(request)
            else:
                request.future.set_result(cached)
        return groups

    def _evaluate_groups(self, groups: OrderedDict[tuple[object, ...], list[_Request]]) -> None:
        unique = [group[0] for group in groups.values()]
        features = [encode_position(request.state) for request in unique]
        batch = InferenceBatch(
            board=torch.from_numpy(np.stack([item.board for item in features])),
            global_features=torch.from_numpy(np.stack([item.global_features for item in features])),
            legal=torch.from_numpy(np.stack([item.legal for item in features])),
        )
        started = time.perf_counter()
        output = self.evaluator.evaluate(batch)
        elapsed = time.perf_counter() - started
        logits = output.policy_logits.detach().cpu().numpy()
        values = output.value.detach().reshape(len(unique)).cpu().numpy()
        ownership = self._optional_numpy(output.ownership)
        scores = self._optional_numpy(output.score_margin, reshape=len(unique))
        if len(logits) != len(unique):
            raise RuntimeError("position evaluator returned the wrong batch size")
        for index, (identity, duplicates) in enumerate(groups.items()):
            evaluation = StateEvaluation(
                policy_logits=np.asarray(logits[index], dtype=np.float32),
                value=float(values[index]),
                ownership=(
                    None if ownership is None else np.asarray(ownership[index], dtype=np.float32)
                ),
                score_margin=None if scores is None else float(scores[index]),
            )
            self._cache_put(identity, evaluation)
            for request in duplicates:
                request.future.set_result(evaluation)
        self._record_batch(len(unique), elapsed)

    @staticmethod
    def _optional_numpy(
        tensor: torch.Tensor | None,
        *,
        reshape: int | None = None,
    ) -> NDArray[np.generic] | None:
        if tensor is None:
            return None
        value = tensor.detach()
        if reshape is not None:
            value = value.reshape(reshape)
        return value.cpu().numpy()

    def _record_batch(self, unique_count: int, elapsed: float) -> None:
        with self._lock:
            self._batches += 1
            self._real_evaluations += unique_count
            self._full_batches += int(unique_count == self.batch_size)
            self._inference_seconds += elapsed
            self._max_inference_seconds = max(self._max_inference_seconds, elapsed)

    @staticmethod
    def _fail_groups(
        groups: OrderedDict[tuple[object, ...], list[_Request]], error: Exception
    ) -> None:
        for duplicates in groups.values():
            for request in duplicates:
                if not request.future.done():
                    request.future.set_exception(error)

    @property
    def stats(self) -> BatchingStats:
        with self._lock:
            slots = self._batches * self.batch_size
            return BatchingStats(
                requests=self._requests,
                cache_hits=self._cache_hits,
                batches=self._batches,
                real_evaluations=self._real_evaluations,
                padded_evaluations=slots - self._real_evaluations,
                full_batches=self._full_batches,
                mean_batch_latency_ms=(
                    0.0 if self._batches == 0 else 1000.0 * self._inference_seconds / self._batches
                ),
                max_batch_latency_ms=1000.0 * self._max_inference_seconds,
                real_batch_fraction=(0.0 if slots == 0 else self._real_evaluations / slots),
                full_batch_fraction=(
                    0.0 if self._batches == 0 else self._full_batches / self._batches
                ),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._thread.join()

    def __enter__(self) -> BatchedInferenceBroker:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
