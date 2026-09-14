"""Pickle-free NPZ codecs for immutable trajectory and annotation records."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from zero_ttt.versioning import RECORD_SCHEMA, SHARD_SCHEMA

from zero_ttt_dataset.records import AnnotationRecord, TrajectoryRecord

NpzArrays = dict[str, NDArray[np.generic]]


def _hex_matrix(values: Sequence[str]) -> NDArray[np.uint8]:
    if not values:
        return np.empty((0, 32), dtype=np.uint8)
    return np.asarray([list(bytes.fromhex(value)) for value in values], dtype=np.uint8)


def _pack_text(prefix: str, values: Sequence[str]) -> NpzArrays:
    encoded = [value.encode("utf-8") for value in values]
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    if encoded:
        offsets[1:] = np.cumsum([len(value) for value in encoded], dtype=np.int64)
    data = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    return {f"{prefix}_data": data, f"{prefix}_offsets": offsets}


def _unpack_text(archive: np.lib.npyio.NpzFile, prefix: str) -> list[str]:
    data = archive[f"{prefix}_data"]
    offsets = archive[f"{prefix}_offsets"]
    return [
        bytes(data[offsets[index] : offsets[index + 1]]).decode("utf-8")
        for index in range(len(offsets) - 1)
    ]


def _concatenate_int[ScalarT: np.generic](
    rows: Sequence[tuple[int, ...]], dtype: np.dtype[ScalarT]
) -> tuple[NDArray[np.int64], NDArray[ScalarT]]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    if rows:
        offsets[1:] = np.cumsum([len(row) for row in rows], dtype=np.int64)
    values = (
        np.asarray([value for row in rows for value in row], dtype=dtype)
        if offsets[-1]
        else np.empty(0, dtype=dtype)
    )
    return offsets, values


class TrajectoryNpzCodec:
    kind_code = 1

    def encode(self, records: Sequence[TrajectoryRecord]) -> NpzArrays:
        if not records:
            raise ValueError("cannot write an empty trajectory shard")
        if any(record.schema_version != RECORD_SCHEMA.current for record in records):
            raise ValueError("new trajectory shards must use the current record schema")
        move_offsets, moves = _concatenate_int(
            [record.moves for record in records], np.dtype(np.int16)
        )
        policy_game_offsets = np.zeros(len(records) + 1, dtype=np.int64)
        policy_game_offsets[1:] = np.cumsum(
            [record.trainable_position_count for record in records], dtype=np.int64
        )
        policy_row_offsets, policy_actions, policy_values = self._policies(records)
        arrays: NpzArrays = {
            "schema_version": np.asarray(SHARD_SCHEMA.current, dtype=np.int32),
            "record_schema_version": np.asarray(RECORD_SCHEMA.current, dtype=np.int32),
            "kind": np.asarray(self.kind_code, dtype=np.uint8),
            "game_ids": _hex_matrix([record.game_id for record in records]),
            "content_hashes": _hex_matrix([record.content_sha256 for record in records]),
            "asset_hashes": _hex_matrix([record.asset_sha256 for record in records]),
            "ordinals": np.asarray([record.ordinal for record in records], dtype=np.int64),
            "komi_half_points": np.asarray(
                [record.komi_half_points for record in records], dtype=np.int16
            ),
            "max_moves": np.asarray([record.max_moves for record in records], dtype=np.int32),
            "move_offsets": move_offsets,
            "moves": moves,
            "trainable_start_ply": np.asarray(
                [record.trainable_start_ply for record in records], dtype=np.int32
            ),
            "policy_game_offsets": policy_game_offsets,
            "policy_row_offsets": policy_row_offsets,
            "policy_actions": policy_actions,
            "policy_values": policy_values,
            "value_black": np.asarray([record.value_black for record in records], dtype=np.float32),
            "value_mask": np.asarray(
                [record.value_available for record in records], dtype=np.bool_
            ),
            "score_margin_black": np.asarray(
                [record.score_margin_black for record in records], dtype=np.float32
            ),
            "score_mask": np.asarray(
                [record.score_available for record in records], dtype=np.bool_
            ),
            "ownership_black": np.asarray(
                [record.ownership_black for record in records], dtype=np.float32
            ),
            "ownership_mask": np.asarray(
                [record.ownership_available for record in records], dtype=np.bool_
            ),
            "game_seeds": np.asarray([record.game_seed for record in records], dtype=np.uint64),
            "search_budgets": self._flat(records, "search_budgets", np.int32),
            "root_values": self._flat(records, "root_values", np.float32),
            "root_score_margins": self._flat(records, "root_score_margins", np.float32),
            "temperatures": self._flat(records, "temperatures", np.float32),
            "search_seeds": self._flat(records, "search_seeds", np.uint64),
            "root_noise_mask": self._flat(records, "root_noise_mask", np.bool_),
            "search_metadata_mask": self._flat(records, "search_metadata_mask", np.bool_),
            "root_score_mask": self._flat(records, "root_score_mask", np.bool_),
        }
        for prefix, field in (
            ("dataset_ids", "dataset_id"),
            ("member_paths", "member_path"),
            ("rules", "rules"),
            ("source_kinds", "source_kind"),
            ("task_ids", "task_id"),
            ("terminations", "termination"),
            ("black_agent_ids", "black_agent_id"),
            ("white_agent_ids", "white_agent_id"),
            ("publication_hashes", "publication_sha256"),
            ("feature_schema_ids", "feature_schema_id"),
            ("search_config_hashes", "search_config_sha256"),
        ):
            arrays.update(_pack_text(prefix, [getattr(record, field) for record in records]))
        return arrays

    @staticmethod
    def _policies(
        records: Sequence[TrajectoryRecord],
    ) -> tuple[NDArray[np.int64], NDArray[np.int16], NDArray[np.float32]]:
        row_offsets = [0]
        actions: list[int] = []
        values: list[float] = []
        for record in records:
            base = row_offsets[-1]
            row_offsets.extend(base + value for value in record.policy_row_offsets[1:])
            actions.extend(record.policy_actions)
            values.extend(record.policy_values)
        return (
            np.asarray(row_offsets, dtype=np.int64),
            np.asarray(actions, dtype=np.int16),
            np.asarray(values, dtype=np.float32),
        )

    @staticmethod
    def _flat[ScalarT: np.generic](
        records: Sequence[TrajectoryRecord], field: str, dtype: type[ScalarT]
    ) -> NDArray[ScalarT]:
        return np.asarray(
            [value for record in records for value in getattr(record, field)],
            dtype=dtype,
        )

    def decode(self, archive: np.lib.npyio.NpzFile) -> tuple[TrajectoryRecord, ...]:
        text = {
            prefix: _unpack_text(archive, prefix)
            for prefix in (
                "dataset_ids",
                "member_paths",
                "rules",
                "source_kinds",
                "task_ids",
                "terminations",
                "black_agent_ids",
                "white_agent_ids",
                "publication_hashes",
                "feature_schema_ids",
                "search_config_hashes",
            )
        }
        return tuple(
            self._decode_record(archive, text, index) for index in range(len(archive["game_ids"]))
        )

    @staticmethod
    def _decode_record(
        archive: np.lib.npyio.NpzFile,
        text: dict[str, list[str]],
        index: int,
    ) -> TrajectoryRecord:
        move_start, move_end = archive["move_offsets"][index : index + 2]
        position_start, position_end = archive["policy_game_offsets"][index : index + 2]
        action_start = archive["policy_row_offsets"][position_start]
        action_end = archive["policy_row_offsets"][position_end]
        row_offsets = archive["policy_row_offsets"][position_start : position_end + 1]
        row_offsets = row_offsets - row_offsets[0]
        position_slice = slice(position_start, position_end)
        action_slice = slice(action_start, action_end)
        return TrajectoryRecord(
            schema_version=int(archive["record_schema_version"]),
            game_id=bytes(archive["game_ids"][index]).hex(),
            content_sha256=bytes(archive["content_hashes"][index]).hex(),
            dataset_id=text["dataset_ids"][index],
            asset_sha256=bytes(archive["asset_hashes"][index]).hex(),
            member_path=text["member_paths"][index],
            ordinal=int(archive["ordinals"][index]),
            rules=text["rules"][index],
            komi_half_points=int(archive["komi_half_points"][index]),
            max_moves=int(archive["max_moves"][index]),
            moves=tuple(int(value) for value in archive["moves"][move_start:move_end]),
            trainable_start_ply=int(archive["trainable_start_ply"][index]),
            policy_row_offsets=tuple(int(value) for value in row_offsets),
            policy_actions=tuple(int(value) for value in archive["policy_actions"][action_slice]),
            policy_values=tuple(float(value) for value in archive["policy_values"][action_slice]),
            value_black=float(archive["value_black"][index]),
            value_available=bool(archive["value_mask"][index]),
            score_margin_black=float(archive["score_margin_black"][index]),
            score_available=bool(archive["score_mask"][index]),
            ownership_black=tuple(float(value) for value in archive["ownership_black"][index]),
            ownership_available=bool(archive["ownership_mask"][index]),
            source_kind=text["source_kinds"][index],
            task_id=text["task_ids"][index],
            termination=text["terminations"][index],
            game_seed=int(archive["game_seeds"][index]),
            black_agent_id=text["black_agent_ids"][index],
            white_agent_id=text["white_agent_ids"][index],
            publication_sha256=text["publication_hashes"][index],
            feature_schema_id=text["feature_schema_ids"][index],
            search_config_sha256=text["search_config_hashes"][index],
            search_budgets=tuple(int(value) for value in archive["search_budgets"][position_slice]),
            root_values=tuple(float(value) for value in archive["root_values"][position_slice]),
            root_score_margins=tuple(
                float(value) for value in archive["root_score_margins"][position_slice]
            ),
            temperatures=tuple(float(value) for value in archive["temperatures"][position_slice]),
            search_seeds=tuple(int(value) for value in archive["search_seeds"][position_slice]),
            root_noise_mask=tuple(
                bool(value) for value in archive["root_noise_mask"][position_slice]
            ),
            search_metadata_mask=tuple(
                bool(value) for value in archive["search_metadata_mask"][position_slice]
            ),
            root_score_mask=tuple(
                bool(value) for value in archive["root_score_mask"][position_slice]
            ),
        )


class AnnotationNpzCodec:
    kind_code = 2

    def encode(self, records: Sequence[AnnotationRecord]) -> NpzArrays:
        if not records:
            raise ValueError("cannot write an empty annotation shard")
        if any(record.schema_version != RECORD_SCHEMA.current for record in records):
            raise ValueError("new annotation shards must use the current record schema")
        policy_offsets, policy_actions = _concatenate_int(
            [record.policy_actions for record in records], np.dtype(np.int16)
        )
        arrays: NpzArrays = {
            "schema_version": np.asarray(SHARD_SCHEMA.current, dtype=np.int32),
            "record_schema_version": np.asarray(RECORD_SCHEMA.current, dtype=np.int32),
            "kind": np.asarray(self.kind_code, dtype=np.uint8),
            "game_ids": _hex_matrix([record.game_id for record in records]),
            "content_hashes": _hex_matrix([record.content_sha256 for record in records]),
            "plies": np.asarray([record.ply for record in records], dtype=np.int32),
            "policy_row_offsets": policy_offsets,
            "policy_actions": policy_actions,
            "policy_values": np.asarray(
                [value for record in records for value in record.policy_values],
                dtype=np.float32,
            ),
            "value": np.asarray([record.value for record in records], dtype=np.float32),
            "value_mask": np.asarray(
                [record.value_available for record in records], dtype=np.bool_
            ),
            "score_margin": np.asarray(
                [record.score_margin for record in records], dtype=np.float32
            ),
            "score_mask": np.asarray(
                [record.score_available for record in records], dtype=np.bool_
            ),
            "ownership": np.asarray([record.ownership for record in records], dtype=np.float32),
            "ownership_mask": np.asarray(
                [record.ownership_available for record in records], dtype=np.bool_
            ),
        }
        arrays.update(
            _pack_text(
                "teacher_fingerprints",
                [record.teacher_fingerprint for record in records],
            )
        )
        return arrays

    def decode(self, archive: np.lib.npyio.NpzFile) -> tuple[AnnotationRecord, ...]:
        teachers = _unpack_text(archive, "teacher_fingerprints")
        records = []
        for index in range(len(archive["game_ids"])):
            start, end = archive["policy_row_offsets"][index : index + 2]
            record = AnnotationRecord(
                schema_version=int(archive["record_schema_version"]),
                game_id=bytes(archive["game_ids"][index]).hex(),
                ply=int(archive["plies"][index]),
                teacher_fingerprint=teachers[index],
                policy_actions=tuple(int(value) for value in archive["policy_actions"][start:end]),
                policy_values=tuple(float(value) for value in archive["policy_values"][start:end]),
                value=float(archive["value"][index]),
                value_available=bool(archive["value_mask"][index]),
                score_margin=float(archive["score_margin"][index]),
                score_available=bool(archive["score_mask"][index]),
                ownership=tuple(float(value) for value in archive["ownership"][index]),
                ownership_available=bool(archive["ownership_mask"][index]),
            )
            stored_hash = bytes(archive["content_hashes"][index]).hex()
            if record.content_sha256 != stored_hash:
                raise ValueError("annotation content_sha256 does not match its contents")
            records.append(record)
        return tuple(records)


def validate_archive(archive: np.lib.npyio.NpzFile, expected_kind: int | None = None) -> None:
    SHARD_SCHEMA.require(_schema_value(archive, "schema_version"))
    RECORD_SCHEMA.require(_schema_value(archive, "record_schema_version"))
    if expected_kind is not None and int(archive["kind"]) != expected_kind:
        raise ValueError("unexpected shard kind")
    for name in archive.files:
        if archive[name].dtype.hasobject:
            raise ValueError(f"object arrays are forbidden in shards: {name}")


def _schema_value(archive: np.lib.npyio.NpzFile, name: str) -> object:
    value = archive[name]
    return value if value.shape != () else value.item()
