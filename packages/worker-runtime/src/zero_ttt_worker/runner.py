"""At-least-once worker loop with cooperative cancellation and lease renewal."""

from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

from zero_ttt_contracts import (
    ArtifactRef,
    CompleteJobRequest,
    DomainEvent,
    EventLevel,
    FailJobRequest,
    HeartbeatRequest,
    JobEnvelope,
    LeaseJobRequest,
    WorkerCapability,
    WorkerRegistration,
)

from zero_ttt_worker.client import ControlClient, ControlClientError

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JobResult:
    result: dict[str, object] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()


class JobContext:
    def __init__(self, client: ControlClient, worker_id: str, job: JobEnvelope) -> None:
        self.client = client
        self.worker_id = worker_id
        self.job = job
        self._cancelled = threading.Event()

    @property
    def cancel_requested(self) -> bool:
        return self._cancelled.is_set()

    def request_cancel(self) -> None:
        self._cancelled.set()

    def emit(
        self,
        kind: str,
        payload: dict[str, object] | None = None,
        *,
        level: EventLevel = EventLevel.INFO,
    ) -> None:
        self.client.event(
            self.job,
            self.worker_id,
            DomainEvent(
                job_id=self.job.job_id,
                kind=kind,
                level=level,
                payload={} if payload is None else payload,
            ),
        )


Handler = Callable[[JobEnvelope, JobContext], JobResult]


class WorkerRunner:
    def __init__(
        self,
        client: ControlClient,
        *,
        worker_id: str,
        capability: WorkerCapability,
        version: str,
        handlers: dict[str, Handler],
        lease_seconds: int = 60,
        idle_seconds: float = 2.0,
    ) -> None:
        self.client = client
        self.worker_id = worker_id
        self.capability = capability
        self.version = version
        self.handlers = handlers
        self.lease_seconds = lease_seconds
        self.idle_seconds = idle_seconds
        self._stop = threading.Event()

    def request_stop(self) -> None:
        self._stop.set()

    def _register(self) -> None:
        self.client.register(
            WorkerRegistration(
                worker_id=self.worker_id,
                capability=self.capability,
                version=self.version,
            )
        )

    def _lease(self) -> JobEnvelope | None:
        return self.client.lease(
            LeaseJobRequest(
                worker_id=self.worker_id,
                capability=self.capability,
                lease_seconds=self.lease_seconds,
            )
        )

    def run_forever(self) -> None:
        registered = False
        while not self._stop.is_set():
            try:
                if not registered:
                    self._register()
                    registered = True
                if self._stop.is_set():
                    break
                job = self._lease()
                if job is None:
                    self._stop.wait(self.idle_seconds)
                    continue
                # A lease already in flight when stopping is still ours to finish.
                self._execute(job)
            except ControlClientError as error:
                logger.warning("Worker %s could not contact Control: %s", self.worker_id, error)
                self._stop.wait(self.idle_seconds)

    def run_once(self) -> bool:
        self._register()
        job = self._lease()
        if job is None:
            return False
        self._execute(job)
        return True

    def _execute(self, job: JobEnvelope) -> None:
        context = JobContext(self.client, self.worker_id, job)
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(job, context, heartbeat_stop),
            name=f"heartbeat-{job.job_id[:8]}",
            daemon=True,
        )
        heartbeat.start()
        try:
            handler = self.handlers.get(job.kind)
            if handler is None:
                raise ValueError(f"worker has no handler for {job.kind}")
            context.emit("job.started", {"attempt": job.attempt, "kind": job.kind})
            result = handler(job, context)
            if context.cancel_requested:
                raise InterruptedError("job stopped at a safe boundary")
            self.client.complete(
                job.job_id,
                CompleteJobRequest(
                    worker_id=self.worker_id,
                    lease_token=job.lease_token,
                    result=result.result,
                    artifacts=result.artifacts,
                ),
            )
        except Exception as error:
            self._report_failure(job, context, error)
        finally:
            heartbeat_stop.set()
            # Wait for any in-flight HTTP renewal to finish before reusing this runner.
            heartbeat.join()

    def _report_failure(self, job: JobEnvelope, context: JobContext, error: Exception) -> None:
        logger.error("Job %s failed", job.job_id, exc_info=error)
        try:
            context.emit(
                "job.failed",
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": "".join(traceback.format_exception(error))[-8000:],
                },
                level=EventLevel.ERROR,
            )
        except ControlClientError as report_error:
            logger.warning("Could not report failure event for %s: %s", job.job_id, report_error)
        try:
            self.client.fail(
                job.job_id,
                FailJobRequest(
                    worker_id=self.worker_id,
                    lease_token=job.lease_token,
                    error_type=type(error).__name__,
                    message=str(error),
                    retryable=not isinstance(error, ValueError | TypeError),
                ),
            )
        except ControlClientError as report_error:
            logger.warning("Could not report failed job %s: %s", job.job_id, report_error)

    def _heartbeat(
        self,
        job: JobEnvelope,
        context: JobContext,
        stop: threading.Event,
    ) -> None:
        interval = max(self.lease_seconds / 3, 1.0)
        while not stop.wait(interval):
            try:
                status = self.client.heartbeat(
                    job.job_id,
                    HeartbeatRequest(
                        worker_id=self.worker_id,
                        lease_token=job.lease_token,
                        lease_seconds=self.lease_seconds,
                    ),
                )
                if status.cancel_requested:
                    context.request_cancel()
            except ControlClientError as error:
                logger.warning("Job %s lost lease renewal: %s", job.job_id, error)
                context.request_cancel()
                return
