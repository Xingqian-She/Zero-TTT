"""Content-addressed shard storage with codec-driven NPZ serialization."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from zero_ttt._io import fsync_directory, sha256_file

from zero_ttt_dataset.codecs import (
    AnnotationNpzCodec,
    TrajectoryNpzCodec,
    validate_archive,
)
from zero_ttt_dataset.records import AnnotationRecord, TrajectoryRecord

ShardKind = Literal["trajectory", "annotation"]
NpzArrays = dict[str, NDArray[np.generic]]


@dataclass(frozen=True, slots=True)
class ShardInfo:
    kind: ShardKind
    sha256: str
    relative_path: str
    size_bytes: int
    record_count: int
    position_count: int


class ShardStore:
    def __init__(
        self,
        root: str | Path,
        *,
        read_only: bool = False,
        trajectory_codec: TrajectoryNpzCodec | None = None,
        annotation_codec: AnnotationNpzCodec | None = None,
    ) -> None:
        self.root = Path(root)
        self.trajectory_dir = self.root / "trajectories"
        self.annotation_dir = self.root / "annotations"
        self.read_only = read_only
        if read_only:
            if not self.trajectory_dir.is_dir() or not self.annotation_dir.is_dir():
                raise FileNotFoundError("read-only shard store is incomplete")
        else:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            self.annotation_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_codec = trajectory_codec or TrajectoryNpzCodec()
        self.annotation_codec = annotation_codec or AnnotationNpzCodec()

    def resolve(self, relative_path: str) -> Path:
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError as error:
            raise ValueError("shard path escapes the store root") from error
        return path

    def write_trajectories(self, records: list[TrajectoryRecord]) -> ShardInfo:
        if self.read_only:
            raise PermissionError("cannot write through a read-only shard store")
        arrays = self.trajectory_codec.encode(records)
        positions = sum(record.trainable_position_count for record in records)
        return self._write("trajectory", arrays, len(records), positions)

    def read_trajectories(
        self, info_or_path: ShardInfo | str | Path
    ) -> tuple[TrajectoryRecord, ...]:
        path = self._coerce_path(info_or_path)
        with self._open_validated(path, expected_kind=self.trajectory_codec.kind_code) as archive:
            return self.trajectory_codec.decode(archive)

    def write_annotations(self, records: list[AnnotationRecord]) -> ShardInfo:
        if self.read_only:
            raise PermissionError("cannot write through a read-only shard store")
        arrays = self.annotation_codec.encode(records)
        return self._write("annotation", arrays, len(records), len(records))

    def read_annotations(
        self, info_or_path: ShardInfo | str | Path
    ) -> tuple[AnnotationRecord, ...]:
        path = self._coerce_path(info_or_path)
        with self._open_validated(path, expected_kind=self.annotation_codec.kind_code) as archive:
            return self.annotation_codec.decode(archive)

    def verify(self, relative_path: str, expected_sha256: str) -> None:
        path = self.resolve(relative_path)
        self._verify_hash(path, expected_sha256, relative_path)
        with self._open_validated(path) as archive:
            kind = int(archive["kind"])
        if kind == self.trajectory_codec.kind_code:
            self.read_trajectories(path)
        elif kind == self.annotation_codec.kind_code:
            self.read_annotations(path)
        else:
            raise ValueError("unexpected shard kind")

    def read_verified_trajectories(
        self, relative_path: str, expected_sha256: str
    ) -> tuple[TrajectoryRecord, ...]:
        path = self.resolve(relative_path)
        self._verify_hash(path, expected_sha256, relative_path)
        return self.read_trajectories(path)

    def read_verified_annotations(
        self, relative_path: str, expected_sha256: str
    ) -> tuple[AnnotationRecord, ...]:
        path = self.resolve(relative_path)
        self._verify_hash(path, expected_sha256, relative_path)
        return self.read_annotations(path)

    @staticmethod
    def _verify_hash(path: Path, expected_sha256: str, relative_path: str) -> None:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"shard SHA-256 mismatch: {relative_path}")

    def _coerce_path(self, value: ShardInfo | str | Path) -> Path:
        if isinstance(value, ShardInfo):
            return self.resolve(value.relative_path)
        path = Path(value)
        return path if path.is_absolute() else self.resolve(path.as_posix())

    def _write(
        self,
        kind: ShardKind,
        arrays: NpzArrays,
        record_count: int,
        position_count: int,
    ) -> ShardInfo:
        directory = self.trajectory_dir if kind == "trajectory" else self.annotation_dir
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".shard-", suffix=".tmp", dir=directory
        )
        temporary = Path(temporary_name)
        try:
            self._save_npz(descriptor, temporary, arrays)
            expected_kind = (
                self.trajectory_codec.kind_code
                if kind == "trajectory"
                else self.annotation_codec.kind_code
            )
            with self._open_validated(temporary, expected_kind=expected_kind):
                pass
            destination, digest = self._commit_content(directory, temporary)
            return ShardInfo(
                kind=kind,
                sha256=digest,
                relative_path=destination.relative_to(self.root).as_posix(),
                size_bytes=destination.stat().st_size,
                record_count=record_count,
                position_count=position_count,
            )
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _save_npz(descriptor: int, temporary: Path, arrays: NpzArrays) -> None:
        with os.fdopen(descriptor, "wb") as handle:
            cast(Any, np.savez)(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _commit_content(directory: Path, temporary: Path) -> tuple[Path, str]:
        digest = sha256_file(temporary)
        destination = directory / f"{digest}.npz"
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError("existing content-addressed shard is corrupt")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
            fsync_directory(directory)
        return destination, digest

    @staticmethod
    def _open_validated(path: Path, expected_kind: int | None = None):
        archive = np.load(path, allow_pickle=False)
        try:
            validate_archive(archive, expected_kind)
        except BaseException:
            archive.close()
            raise
        return archive
