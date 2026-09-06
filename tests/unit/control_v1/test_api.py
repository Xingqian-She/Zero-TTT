from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from zero_ttt_contracts import HeartbeatRequest, LeaseJobRequest, WorkerCapability, WorkflowTemplate
from zero_ttt_control.api import create_app
from zero_ttt_control.store import ControlStore


def test_openapi_and_strict_workflow_request(tmp_path) -> None:
    client = TestClient(create_app(ControlStore(tmp_path / "control.sqlite")))
    assert client.get("/healthz").json() == {"ok": True}
    document = client.get("/openapi.json").json()
    assert "/internal/v1/jobs/lease" in document["paths"]
    response = client.post(
        "/api/v1/workflows/data-bootstrap",
        json={"parameters": {}, "unexpected": True},
    )
    assert response.status_code == 422


def test_http_errors_preserve_status_and_hide_unexpected_details(tmp_path, monkeypatch, caplog):
    store = ControlStore(tmp_path / "control.sqlite")
    with TestClient(create_app(store, close_store=True), raise_server_exceptions=False) as client:
        assert client.get("/api/v1/jobs/missing").status_code == 404
        workflow_id = store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {})
        job_id = store.list_jobs(workflow_id)[0]["job_id"]
        assert client.post(f"/api/v1/jobs/{job_id}/retry").status_code == 422
        job = store.lease_job(LeaseJobRequest(worker_id="data", capability=WorkerCapability.DATA))
        assert job is not None
        response = client.post(
            f"/internal/v1/jobs/{job_id}/heartbeat",
            json=HeartbeatRequest(worker_id="wrong", lease_token="wrong").model_dump(),
        )
        assert response.status_code == 409

        def unexpected():
            raise RuntimeError("private database location")

        monkeypatch.setattr(store, "snapshot", unexpected)
        response = client.get("/api/v1/snapshot")
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal Server Error"}
        assert "private database location" in caplog.text


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "test", "profile_id": "test", "cold_snapshot_id": "missing"},
        {"name": "test", "profile_id": "test", "cold_snapshot_id": "missing", "unknown": True},
    ],
)
def test_create_run_errors_keep_request_validation_and_not_found_status(tmp_path, payload):
    with TestClient(
        create_app(ControlStore(tmp_path / "control.sqlite"), close_store=True)
    ) as client:
        response = client.post("/api/v1/runs", json=payload)
        assert response.status_code == (422 if "unknown" in payload else 404)
