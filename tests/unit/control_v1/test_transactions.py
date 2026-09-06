from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from zero_ttt_contracts import (
    ArtifactKind,
    ArtifactRef,
    CompleteJobRequest,
    DomainEvent,
    FailJobRequest,
    HeartbeatRequest,
    JobState,
    LeaseJobRequest,
    RunSpec,
    WorkerCapability,
    WorkflowTemplate,
)
from zero_ttt_control.store import ControlStore, LeaseConflict


@pytest.fixture
def store(tmp_path):
    instance = ControlStore(tmp_path / "control.sqlite")
    yield instance
    instance.close()


def crash(*_args):
    raise RuntimeError("injected workflow update failure")


@pytest.mark.parametrize("operation", ["cancel", "retry"])
def test_job_and_workflow_updates_roll_back_together(store, monkeypatch, operation):
    workflow_id = store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {})
    job_id = store.list_jobs(workflow_id)[0]["job_id"]
    if operation == "retry":
        store.cancel(job_id)
    before = store.snapshot()
    monkeypatch.setattr(store, "_refresh_workflow", crash)
    with pytest.raises(RuntimeError, match="injected"):
        getattr(store, operation)(job_id)
    assert store.snapshot() == before
    assert not store.connection.in_transaction


@pytest.mark.parametrize("operation", ["complete", "fail"])
def test_terminal_update_rolls_back_artifacts_and_resource_release(store, monkeypatch, operation):
    store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {})
    job = store.lease_job(LeaseJobRequest(worker_id="data", capability=WorkerCapability.DATA))
    assert job is not None
    before = store.snapshot()
    monkeypatch.setattr(store, "_refresh_workflow", crash)
    if operation == "complete":
        request = CompleteJobRequest(
            worker_id="data",
            lease_token=job.lease_token,
            artifacts=(
                ArtifactRef(
                    kind=ArtifactKind.DATA_VERIFICATION,
                    artifact_id="verification.test",
                    format_version=1,
                    sha256="a" * 64,
                    uri="artifact://data/test.json",
                    size_bytes=1,
                ),
            ),
        )
    else:
        request = FailJobRequest(
            worker_id="data",
            lease_token=job.lease_token,
            error_type="ValueError",
            message="invalid input",
            retryable=False,
        )
    with pytest.raises(RuntimeError, match="injected"):
        getattr(store, operation)(job.job_id, request)
    assert store.snapshot() == before
    resource = store.connection.execute("SELECT job_id FROM resource_leases").fetchone()
    assert resource["job_id"] == job.job_id
    assert not store.connection.in_transaction


@pytest.mark.parametrize("separate_connections", [False, True])
def test_concurrent_same_key_submissions_return_one_workflow(store, separate_connections):
    stores = [store]
    if separate_connections:
        stores.extend(ControlStore(store.path) for _ in range(3))
    barrier = threading.Barrier(4)

    def submit(index):
        barrier.wait(timeout=5)
        return stores[index % len(stores)].submit_workflow(
            WorkflowTemplate.DATA_BOOTSTRAP, {"trial_games": 4}, idempotency_key="same-key"
        )

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            workflow_ids = list(pool.map(submit, range(4)))
        assert len(set(workflow_ids)) == 1
        assert len(store.list_workflows()) == 1
        assert len(store.list_jobs()) == 7
        with pytest.raises(ValueError, match="different content"):
            store.submit_workflow(
                WorkflowTemplate.DATA_BOOTSTRAP, {"trial_games": 5}, idempotency_key="same-key"
            )
        assert len(store.list_workflows()) == 1
    finally:
        for instance in stores[1:]:
            instance.close()


def test_expired_owner_cannot_renew_or_complete_reclaimed_job(store):
    clock = [1_000_000_000]
    store.clock_ns = lambda: clock[0]
    store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {})
    first = store.lease_job(
        LeaseJobRequest(worker_id="old", capability=WorkerCapability.DATA, lease_seconds=10)
    )
    assert first is not None
    clock[0] += 11_000_000_000
    second = store.lease_job(
        LeaseJobRequest(worker_id="new", capability=WorkerCapability.DATA, lease_seconds=10)
    )
    assert second is not None and second.job_id == first.job_id and second.attempt == 2
    with pytest.raises(LeaseConflict):
        store.heartbeat(
            first.job_id, HeartbeatRequest(worker_id="old", lease_token=first.lease_token)
        )
    with pytest.raises(LeaseConflict):
        store.complete(
            first.job_id, CompleteJobRequest(worker_id="old", lease_token=first.lease_token)
        )
    assert store.get_job(second.job_id)["state"] == JobState.LEASED.value


def test_concurrent_event_retries_preserve_one_sequence(store):
    store.submit_workflow(WorkflowTemplate.DATA_BOOTSTRAP, {})
    job = store.lease_job(LeaseJobRequest(worker_id="data", capability=WorkerCapability.DATA))
    assert job is not None
    event = DomainEvent(job_id=job.job_id, kind="test.progress")
    other = ControlStore(store.path)
    barrier = threading.Barrier(2)

    def append(instance):
        barrier.wait(timeout=5)
        return instance.append_event(job.job_id, "data", job.lease_token, event)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sequences = list(pool.map(append, (store, other)))
        assert sequences[0] == sequences[1]
        with pytest.raises(ValueError, match="different content"):
            store.append_event(
                job.job_id,
                "data",
                job.lease_token,
                event.model_copy(update={"payload": {"changed": True}}),
            )
        assert len(store.events()) == 1
    finally:
        other.close()


def test_gpu_claims_are_atomic_and_failure_releases_resource(store):
    run = RunSpec(
        run_id="run",
        name="run",
        profile_id="test",
        profile_sha256="a" * 64,
        profile={},
        cold_snapshot=ArtifactRef(
            kind=ArtifactKind.DATASET_SNAPSHOT,
            artifact_id="dataset.test",
            format_version=1,
            sha256="b" * 64,
            uri="artifact://data/test.json",
            size_bytes=1,
        ),
    )
    store.create_run(run)
    for key in ("one", "two"):
        store.submit_workflow(
            WorkflowTemplate.COLD_START, {}, run_id=run.run_id, idempotency_key=key
        )
    barrier = threading.Barrier(2)

    def claim(worker_id):
        barrier.wait(timeout=5)
        return worker_id, store.lease_job(
            LeaseJobRequest(worker_id=worker_id, capability=WorkerCapability.TRAINER)
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = [(owner, job) for owner, job in pool.map(claim, ("one", "two")) if job]
    assert len(claimed) == 1
    owner, job = claimed[0]
    store.fail(
        job.job_id,
        FailJobRequest(
            worker_id=owner,
            lease_token=job.lease_token,
            error_type="ValueError",
            message="invalid",
            retryable=False,
        ),
    )
    next_job = store.lease_job(
        LeaseJobRequest(worker_id="next", capability=WorkerCapability.TRAINER)
    )
    assert next_job is not None and next_job.job_id != job.job_id
