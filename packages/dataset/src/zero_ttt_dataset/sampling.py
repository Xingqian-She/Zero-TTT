"""Immutable snapshot position indexing and shard-local sampling."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zero_ttt_dataset.locators import AnnotationLocator, TrajectoryLocator

AnnotationMode = Literal["none", "prefer_exact", "require_exact"]


@dataclass(frozen=True, slots=True)
class SampleReference:
    trajectory: TrajectoryLocator
    local_position: int
    annotation: AnnotationLocator | None


@dataclass(frozen=True, slots=True)
class _TrajectoryBucket:
    shard_sha256: str
    trajectories: tuple[TrajectoryLocator, ...]
    cumulative_positions: NDArray[np.int64]
    position_count: int


@dataclass(frozen=True, slots=True)
class _ExactBucket:
    shard_sha256: str
    references: tuple[SampleReference, ...]
    position_count: int


_PositionBucket = _TrajectoryBucket | _ExactBucket


class SnapshotPositionIndex:
    """Pure, immutable position index with weighted shard-local sampling."""

    def __init__(
        self,
        trajectories: tuple[TrajectoryLocator, ...],
        annotations: tuple[AnnotationLocator, ...],
        annotation_mode: AnnotationMode,
    ) -> None:
        if not trajectories:
            raise ValueError("snapshot has no trajectories")
        annotation_by_position = {
            (locator.game_id, locator.ply): locator for locator in annotations
        }
        if annotation_mode == "require_exact":
            buckets = self._exact_buckets(trajectories, annotations)
        else:
            buckets = self._trajectory_buckets(trajectories)
        if not buckets:
            if annotation_mode == "require_exact":
                raise ValueError("snapshot has no positions for the required teacher")
            raise ValueError("snapshot has no trainable positions")
        self._buckets = tuple(buckets)
        self._bucket_cumulative = np.cumsum(
            [bucket.position_count for bucket in buckets], dtype=np.int64
        )
        self.position_count = int(self._bucket_cumulative[-1])
        self.annotation_mode = annotation_mode
        self._annotation_by_position = annotation_by_position

    @staticmethod
    def _trajectory_buckets(
        trajectories: tuple[TrajectoryLocator, ...],
    ) -> list[_PositionBucket]:
        grouped: OrderedDict[str, list[TrajectoryLocator]] = OrderedDict()
        for trajectory in trajectories:
            if trajectory.trainable_positions:
                grouped.setdefault(trajectory.shard_sha256, []).append(trajectory)
        buckets: list[_PositionBucket] = []
        for shard_sha256, entries in grouped.items():
            shard_trajectories = tuple(entries)
            cumulative = np.cumsum(
                [locator.trainable_positions for locator in shard_trajectories],
                dtype=np.int64,
            )
            buckets.append(
                _TrajectoryBucket(
                    shard_sha256,
                    shard_trajectories,
                    cumulative,
                    int(cumulative[-1]),
                )
            )
        return buckets

    @staticmethod
    def _exact_buckets(
        trajectories: tuple[TrajectoryLocator, ...],
        annotations: tuple[AnnotationLocator, ...],
    ) -> list[_PositionBucket]:
        trajectory_by_game = {locator.game_id: locator for locator in trajectories}
        grouped: OrderedDict[str, list[SampleReference]] = OrderedDict()
        for annotation in annotations:
            trajectory = trajectory_by_game.get(annotation.game_id)
            if trajectory is None:
                continue
            end_ply = trajectory.trainable_start_ply + trajectory.trainable_positions
            if trajectory.trainable_start_ply <= annotation.ply < end_ply:
                grouped.setdefault(trajectory.shard_sha256, []).append(
                    SampleReference(
                        trajectory,
                        annotation.ply - trajectory.trainable_start_ply,
                        annotation,
                    )
                )
        return [
            _ExactBucket(shard_sha256, tuple(references), len(references))
            for shard_sha256, references in grouped.items()
            if references
        ]

    def draw_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[SampleReference, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shard_draw = int(rng.integers(0, self.position_count))
        bucket_index = int(np.searchsorted(self._bucket_cumulative, shard_draw, side="right"))
        bucket = self._buckets[bucket_index]
        draws = rng.integers(0, bucket.position_count, size=batch_size)
        if isinstance(bucket, _ExactBucket):
            return tuple(bucket.references[int(draw)] for draw in draws)
        return tuple(self._reference_for_draw(bucket, int(draw)) for draw in draws)

    def _reference_for_draw(
        self,
        bucket: _TrajectoryBucket,
        draw: int,
    ) -> SampleReference:
        trajectory_index = int(np.searchsorted(bucket.cumulative_positions, draw, side="right"))
        previous = (
            0 if trajectory_index == 0 else int(bucket.cumulative_positions[trajectory_index - 1])
        )
        trajectory = bucket.trajectories[trajectory_index]
        local_position = draw - previous
        ply = trajectory.trainable_start_ply + local_position
        annotation = self._annotation_by_position.get((trajectory.game_id, ply))
        return SampleReference(trajectory, local_position, annotation)
