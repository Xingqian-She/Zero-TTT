"""Control-owned SQLite connection, locking, and atomic write boundaries."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

_SCHEMA_VERSION = 1


class ControlSession:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA synchronous=FULL")
            self._initialize()
        except BaseException:
            self.connection.close()
            raise

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    @contextmanager
    def read(self) -> Generator[sqlite3.Connection, None, None]:
        # Consumers must materialize cursors before releasing the shared connection.
        with self._lock:
            yield self.connection

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def _initialize(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in {0, _SCHEMA_VERSION}:
            raise RuntimeError(f"unsupported control database schema v{version}")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                capability TEXT NOT NULL,
                version TEXT NOT NULL,
                registered_ns INTEGER NOT NULL,
                last_seen_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                spec_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL DEFAULT '',
                template TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id),
                run_id TEXT NOT NULL DEFAULT '',
                ordinal INTEGER NOT NULL,
                kind TEXT NOT NULL,
                capability TEXT NOT NULL,
                resource_class TEXT NOT NULL,
                state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                attempt INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                lease_owner TEXT NOT NULL DEFAULT '',
                lease_token TEXT NOT NULL DEFAULT '',
                lease_expires_ns INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                result_json TEXT NOT NULL DEFAULT '{}',
                error TEXT NOT NULL DEFAULT '',
                created_ns INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL,
                UNIQUE(workflow_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS job_dependencies (
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                depends_on_job_id TEXT NOT NULL REFERENCES jobs(job_id),
                PRIMARY KEY(job_id, depends_on_job_id)
            );
            CREATE TABLE IF NOT EXISTS resource_leases (
                resource_class TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                lease_token TEXT NOT NULL,
                expires_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                kind TEXT NOT NULL,
                ref_json TEXT NOT NULL,
                created_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                kind TEXT NOT NULL,
                level TEXT NOT NULL,
                occurred_ns INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS jobs_claim_idx
              ON jobs(capability,state,ordinal,created_ns);
            CREATE INDEX IF NOT EXISTS events_job_idx ON events(job_id,sequence);
            """
        )
        self.connection.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
