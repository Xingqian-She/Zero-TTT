from __future__ import annotations

import json

import pytest
import zero_ttt._io as io_helpers
from zero_ttt._io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


def test_canonical_json_and_hashes_are_deterministic(tmp_path) -> None:
    first = canonical_json_bytes({"b": 2, "a": "围棋"})
    second = canonical_json_bytes({"a": "围棋", "b": 2})
    assert first == second == b'{"a":"\xe5\x9b\xb4\xe6\xa3\x8b","b":2}'
    path = tmp_path / "payload.bin"
    atomic_write_bytes(path, first)
    assert sha256_file(path) == sha256_bytes(first)


def test_atomic_json_replaces_the_destination(tmp_path) -> None:
    path = tmp_path / "nested" / "state.json"
    atomic_write_json(path, {"version": 1})
    atomic_write_json(path, {"version": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}
    assert list(path.parent.glob(".tmp-*")) == []


def test_atomic_write_preserves_destination_when_replace_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    atomic_write_bytes(path, b"old")

    def fail_replace(_source, _destination) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(io_helpers.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_write_bytes(path, b"new")
    assert path.read_bytes() == b"old"
    assert list(tmp_path.glob(".tmp-*")) == []
