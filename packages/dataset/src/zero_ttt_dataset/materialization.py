"""Snapshot-bound, shard-local training batches from cataloged records."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from zero_ttt.config import GameConfig
from zero_ttt.game.features import GLOBAL_FEATURES, POINT_FEATURES, encode_position
from zero_ttt.game.rules import ACTION_SIZE, BOARD_AREA, BOARD_SIZE, Color
from zero_ttt.game.state import GameState
from zero_ttt.game.symmetry import augment_sample

from zero_ttt_dataset.contracts import TrainBatch
from zero_ttt_dataset.locators import AnnotationLocator, TrajectoryLocator
from zero_ttt_dataset.records import AnnotationRecord, TrajectoryRecord
from zero_ttt_dataset.sampling import SampleReference
from zero_ttt_dataset.shards import ShardStore

RecordT = TypeVar("RecordT", TrajectoryRecord, AnnotationRecord)


class ShardRecordCache:
    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("shard_cache_size must be positive")
        self.max_size = max_size
        self._values: OrderedDict[tuple[str, str], tuple[object, ...]] = OrderedDict()

    def get(
        self,
        kind: str,
        sha256: str,
        loader: Callable[[], tuple[RecordT, ...]],
    ) -> tuple[RecordT, ...]:
        key = (kind, sha256)
        cached = self._values.get(key)
        if cached is None:
            cached = loader()
            self._values[key] = cached
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)
        self._values.move_to_end(key)
        return cast(tuple[RecordT, ...], cached)


@dataclass(slots=True)
class _BatchArrays:
    boards: NDArray[np.float32]
    globals: NDArray[np.float32]
    legal: NDArray[np.bool_]
    policies: NDArray[np.float32]
    values: NDArray[np.float32]
    ownerships: NDArray[np.float32]
    scores: NDArray[np.float32]
    value_masks: NDArray[np.bool_]
    ownership_masks: NDArray[np.bool_]
    score_masks: NDArray[np.bool_]

    @classmethod
    def allocate(cls, batch_size: int) -> _BatchArrays:
        return cls(
            np.empty((batch_size, POINT_FEATURES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            np.empty((batch_size, GLOBAL_FEATURES), dtype=np.float32),
            np.empty((batch_size, ACTION_SIZE), dtype=np.bool_),
            np.zeros((batch_size, ACTION_SIZE), dtype=np.float32),
            np.empty(batch_size, dtype=np.float32),
            np.empty((batch_size, BOARD_AREA), dtype=np.float32),
            np.empty(batch_size, dtype=np.float32),
            np.empty(batch_size, dtype=np.bool_),
            np.empty(batch_size, dtype=np.bool_),
            np.empty(batch_size, dtype=np.bool_),
        )

    def as_batch(self) -> TrainBatch:
        return TrainBatch(
            board=self.boards,
            global_features=self.globals,
            legal=self.legal,
            policy=self.policies,
            value=self.values,
            ownership=self.ownerships,
            score_margin=self.scores,
            value_mask=self.value_masks,
            ownership_mask=self.ownership_masks,
            score_mask=self.score_masks,
        )


@dataclass(slots=True)
class _SampleTargets:
    policy: NDArray[np.float32]
    value: np.float32
    ownership: NDArray[np.float32]
    score: np.float32
    value_mask: bool
    ownership_mask: bool
    score_mask: bool


class TrajectoryBatchMaterializer:
    """Load sampled shards once and turn logical records into a TrainBatch."""

    def __init__(
        self,
        store: ShardStore,
        shard_cache_size: int,
        cache: ShardRecordCache | None = None,
    ) -> None:
        self.store = store
        self.cache = cache or ShardRecordCache(shard_cache_size)

    def _trajectory_shard(
        self,
        locator: TrajectoryLocator,
    ) -> tuple[TrajectoryRecord, ...]:
        return self.cache.get(
            "trajectory",
            locator.shard_sha256,
            lambda: self.store.read_verified_trajectories(
                locator.relative_path,
                locator.shard_sha256,
            ),
        )

    def _annotation_shard(
        self,
        locator: AnnotationLocator,
    ) -> tuple[AnnotationRecord, ...]:
        return self.cache.get(
            "annotation",
            locator.shard_sha256,
            lambda: self.store.read_verified_annotations(
                locator.relative_path,
                locator.shard_sha256,
            ),
        )

    @staticmethod
    def _trajectory_record(
        records: tuple[TrajectoryRecord, ...],
        locator: TrajectoryLocator,
    ) -> TrajectoryRecord:
        try:
            record = records[locator.game_index]
        except IndexError as error:
            raise ValueError("catalog game_index is outside its shard") from error
        if record.game_id != locator.game_id or record.content_sha256 != locator.content_sha256:
            raise ValueError("catalog and trajectory shard identities disagree")
        return record

    @staticmethod
    def _annotation_record(
        records: tuple[AnnotationRecord, ...],
        locator: AnnotationLocator,
    ) -> AnnotationRecord:
        try:
            record = records[locator.record_index]
        except IndexError as error:
            raise ValueError("catalog annotation index is outside its shard") from error
        if (
            record.game_id != locator.game_id
            or record.content_sha256 != locator.content_sha256
            or record.ply != locator.ply
            or record.teacher_fingerprint != locator.teacher_fingerprint
        ):
            raise ValueError("catalog and annotation shard identities disagree")
        return record

    @staticmethod
    def _initial_state(record: TrajectoryRecord) -> GameState:
        return GameState.new(
            GameConfig(
                board_size=BOARD_SIZE,
                komi_half_points=record.komi_half_points,
                max_moves=record.max_moves,
                history_length=8,
            )
        )

    def _replay_references(
        self,
        references: tuple[SampleReference, ...],
        trajectory_records: tuple[TrajectoryRecord, ...],
    ) -> tuple[tuple[TrajectoryRecord, ...], tuple[GameState, ...]]:
        selected = tuple(
            self._trajectory_record(trajectory_records, reference.trajectory)
            for reference in references
        )
        grouped: dict[str, list[tuple[int, int]]] = {}
        for index, (reference, record) in enumerate(zip(references, selected, strict=True)):
            ply = record.trainable_start_ply + reference.local_position
            grouped.setdefault(record.game_id, []).append((index, ply))

        states: list[GameState | None] = [None] * len(references)
        for entries in grouped.values():
            record = selected[entries[0][0]]
            state = self._initial_state(record)
            current_ply = 0
            by_ply: dict[int, list[int]] = {}
            for index, ply in entries:
                by_ply.setdefault(ply, []).append(index)
            for ply, indexes in sorted(by_ply.items()):
                state = state.play_many(record.moves[current_ply:ply])
                current_ply = ply
                for index in indexes:
                    states[index] = state
        if any(state is None for state in states):
            raise RuntimeError("failed to materialize a sampled game state")
        return selected, cast(tuple[GameState, ...], tuple(states))

    def materialize(
        self,
        references: tuple[SampleReference, ...],
        rng: np.random.Generator,
    ) -> TrainBatch:
        if not references:
            raise ValueError("cannot materialize an empty batch")
        shard_ids = {reference.trajectory.shard_sha256 for reference in references}
        if len(shard_ids) != 1:
            raise ValueError("a shard-local microbatch must reference exactly one trajectory shard")
        trajectory_records = self._trajectory_shard(references[0].trajectory)
        selected_records, states = self._replay_references(references, trajectory_records)
        annotation_shards: dict[str, tuple[AnnotationRecord, ...]] = {}
        for reference in references:
            locator = reference.annotation
            if locator is not None and locator.shard_sha256 not in annotation_shards:
                annotation_shards[locator.shard_sha256] = self._annotation_shard(locator)
        arrays = _BatchArrays.allocate(len(references))
        for index, reference in enumerate(references):
            targets = self._sample_targets(
                selected_records[index],
                reference,
                states[index],
                annotation_shards,
            )
            self._store_augmented(arrays, index, states[index], targets, rng)
        return arrays.as_batch()

    def _sample_targets(
        self,
        record: TrajectoryRecord,
        reference: SampleReference,
        state: GameState,
        annotation_shards: dict[str, tuple[AnnotationRecord, ...]],
    ) -> _SampleTargets:
        actions, masses = record.policy_at(reference.local_position)
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        policy[np.asarray(actions, dtype=np.int64)] = np.asarray(masses, dtype=np.float32)
        perspective = 1.0 if state.to_play is Color.BLACK else -1.0
        targets = _SampleTargets(
            policy=policy,
            value=np.float32(record.value_black * perspective),
            ownership=np.asarray(record.ownership_black, dtype=np.float32) * perspective,
            score=np.float32(record.score_margin_black * perspective),
            value_mask=record.value_available,
            ownership_mask=record.ownership_available,
            score_mask=record.score_available,
        )
        locator = reference.annotation
        if locator is None:
            return targets
        annotation = self._annotation_record(annotation_shards[locator.shard_sha256], locator)
        if annotation.policy_values:
            targets.policy.fill(0.0)
            targets.policy[np.asarray(annotation.policy_actions, dtype=np.int64)] = np.asarray(
                annotation.policy_values, dtype=np.float32
            )
        if annotation.value_available:
            targets.value, targets.value_mask = np.float32(annotation.value), True
        if annotation.score_available:
            targets.score, targets.score_mask = np.float32(annotation.score_margin), True
        if annotation.ownership_available:
            targets.ownership = np.asarray(annotation.ownership, dtype=np.float32)
            targets.ownership_mask = True
        return targets

    @staticmethod
    def _store_augmented(
        arrays: _BatchArrays,
        index: int,
        state: GameState,
        targets: _SampleTargets,
        rng: np.random.Generator,
    ) -> None:
        augmented = augment_sample(
            encode_position(state),
            targets.policy,
            targets.ownership,
            int(rng.integers(0, 8)),
        )
        arrays.boards[index] = augmented.features.board
        arrays.globals[index] = augmented.features.global_features
        arrays.legal[index] = augmented.features.legal
        arrays.policies[index] = augmented.policy
        arrays.values[index] = targets.value
        arrays.ownerships[index] = augmented.ownership
        arrays.scores[index] = targets.score
        arrays.value_masks[index] = targets.value_mask
        arrays.ownership_masks[index] = targets.ownership_mask
        arrays.score_masks[index] = targets.score_mask
