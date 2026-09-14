"""Public and worker HTTP surfaces for the durable control plane."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from zero_ttt_contracts import (
    CompleteJobRequest,
    DomainEvent,
    FailJobRequest,
    HeartbeatRequest,
    LeaseJobRequest,
    RunSpec,
    WorkerRegistration,
    WorkflowTemplate,
)
from zero_ttt_contracts.hashing import payload_sha256

from zero_ttt_control.profiles import ProfileRepository
from zero_ttt_control.store import ControlStore, LeaseConflict


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateRunRequest(RequestModel):
    name: str = Field(min_length=1, max_length=80)
    profile_id: str
    cold_snapshot_id: str


class SubmitWorkflowRequest(RequestModel):
    run_id: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


async def _handle_error(request: Request, error: Exception) -> JSONResponse:
    if isinstance(error, KeyError):
        status = 404
    elif isinstance(error, LeaseConflict):
        status = 409
    elif isinstance(error, ValueError | sqlite3.IntegrityError):
        status = 422
    else:
        logging.getLogger(__name__).error(
            "Unhandled Control error: %s %s", request.method, request.url.path, exc_info=error
        )
        return JSONResponse({"detail": "Internal Server Error"}, status_code=500)
    return JSONResponse({"detail": str(error)}, status_code=status)


def create_app(  # noqa: C901 - route assembly keeps API dependencies explicit
    store: ControlStore,
    *,
    profile_root: str | Path = "configs/profiles",
    close_store: bool = False,
) -> FastAPI:
    profiles = ProfileRepository(profile_root)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if close_store:
                store.close()

    app = FastAPI(title="Zero-TTT Control API", version="1.0.0", lifespan=lifespan)
    for error_type in (KeyError, LeaseConflict, ValueError, sqlite3.IntegrityError, Exception):
        app.add_exception_handler(error_type, _handle_error)

    @app.get("/healthz")
    def health() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/v1/snapshot")
    def snapshot() -> dict[str, Any]:
        return store.snapshot()

    @app.get("/api/v1/runs")
    def runs() -> dict[str, Any]:
        return {"runs": [item.model_dump(mode="json") for item in store.list_runs()]}

    @app.get("/api/v1/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        return store.get_run(run_id).model_dump(mode="json")

    @app.get("/api/v1/profiles")
    def list_profiles() -> dict[str, Any]:
        return {"profiles": profiles.list()}

    @app.post("/api/v1/runs", status_code=201)
    def create_run(request: CreateRunRequest) -> dict[str, Any]:
        try:
            cold = store.get_artifact(request.cold_snapshot_id, kind="dataset-snapshot")
        except KeyError as error:
            raise KeyError(f"unknown snapshot {request.cold_snapshot_id}") from error
        if cold.labels.get("split") != "train":
            raise ValueError("cold-start requires a train snapshot")
        profile = profiles.load(request.profile_id)
        profile_sha = payload_sha256(profile)
        spec = RunSpec(
            run_id=uuid.uuid4().hex,
            name=request.name.strip(),
            profile_id=request.profile_id,
            profile_sha256=profile_sha,
            profile=profile,
            cold_snapshot=cold,
        )
        return store.create_run(spec).model_dump(mode="json")

    @app.get("/api/v1/workflows")
    def workflows() -> dict[str, Any]:
        return {"workflows": store.list_workflows()}

    @app.get("/api/v1/workflows/{workflow_id}")
    def workflow(workflow_id: str) -> dict[str, Any]:
        return store.get_workflow(workflow_id)

    @app.post("/api/v1/workflows/{template}", status_code=202)
    def submit_workflow(
        template: WorkflowTemplate, request: SubmitWorkflowRequest
    ) -> dict[str, str]:
        workflow_id = store.submit_workflow(
            template,
            request.parameters,
            run_id=request.run_id,
            idempotency_key=request.idempotency_key,
        )
        return {"workflow_id": workflow_id}

    @app.get("/api/v1/jobs")
    def jobs(workflow_id: str | None = None) -> dict[str, Any]:
        return {"jobs": store.list_jobs(workflow_id)}

    @app.get("/api/v1/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        return store.get_job(job_id)

    @app.post("/api/v1/jobs/{job_id}/cancel", status_code=202)
    def cancel(job_id: str) -> dict[str, str]:
        store.cancel(job_id)
        return {"job_id": job_id}

    @app.post("/api/v1/jobs/{job_id}/retry", status_code=202)
    def retry(job_id: str) -> dict[str, str]:
        store.retry(job_id)
        return {"job_id": job_id}

    @app.get("/api/v1/artifacts")
    def artifacts(kind: str | None = None) -> dict[str, Any]:
        return {"artifacts": [item.model_dump(mode="json") for item in store.list_artifacts(kind)]}

    @app.get("/api/v1/datasets")
    def datasets() -> dict[str, Any]:
        return {
            "datasets": [
                item.model_dump(mode="json") for item in store.list_artifacts("dataset-snapshot")
            ]
        }

    @app.get("/api/v1/datasets/{artifact_id}")
    def dataset(artifact_id: str) -> dict[str, Any]:
        return store.get_artifact(artifact_id, kind="dataset-snapshot").model_dump(mode="json")

    @app.get("/api/v1/publications")
    def publications() -> dict[str, Any]:
        return {
            "publications": [
                item.model_dump(mode="json") for item in store.list_artifacts("publication")
            ]
        }

    @app.get("/api/v1/publications/{artifact_id}")
    def publication(artifact_id: str) -> dict[str, Any]:
        return store.get_artifact(artifact_id, kind="publication").model_dump(mode="json")

    @app.get("/api/v1/events")
    def events(after: int = 0, limit: int = Query(500, ge=1, le=2000)) -> dict[str, Any]:
        return {"events": store.events(after, limit)}

    @app.get("/api/v1/events/stream")
    async def event_stream(after: int = 0) -> StreamingResponse:
        async def generate() -> AsyncIterator[str]:
            cursor = after
            while True:
                rows = store.events(cursor, 500)
                if rows:
                    for row in rows:
                        cursor = int(row["sequence"])
                        yield f"id: {cursor}\ndata: {json.dumps(row, ensure_ascii=False)}\n\n"
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/internal/v1/workers/register")
    def register_worker(request: WorkerRegistration) -> dict[str, Any]:
        return store.register_worker(request)

    @app.post("/internal/v1/jobs/lease")
    async def lease_job(request: LeaseJobRequest) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + request.wait_seconds
        while True:
            job = store.lease_job(request)
            if job is not None or asyncio.get_running_loop().time() >= deadline:
                return {"job": None if job is None else job.model_dump(mode="json")}
            await asyncio.sleep(0.25)

    @app.post("/internal/v1/jobs/{job_id}/heartbeat")
    def heartbeat(job_id: str, request: HeartbeatRequest) -> dict[str, Any]:
        return store.heartbeat(job_id, request).model_dump(mode="json")

    @app.post("/internal/v1/jobs/{job_id}/events", status_code=202)
    def append_event(
        job_id: str,
        event: DomainEvent,
        x_worker_id: str = Header(),
        x_lease_token: str = Header(),
    ) -> dict[str, int]:
        return {"sequence": store.append_event(job_id, x_worker_id, x_lease_token, event)}

    @app.post("/internal/v1/jobs/{job_id}/complete", status_code=202)
    def complete(job_id: str, request: CompleteJobRequest) -> dict[str, str]:
        store.complete(job_id, request)
        return {"job_id": job_id}

    @app.post("/internal/v1/jobs/{job_id}/fail", status_code=202)
    def fail(job_id: str, request: FailJobRequest) -> dict[str, str]:
        store.fail(job_id, request)
        return {"job_id": job_id}

    return app
