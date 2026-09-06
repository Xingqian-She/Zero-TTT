from __future__ import annotations

import signal
import threading
from unittest.mock import create_autospec

import pytest
from zero_ttt_contracts import HeartbeatResponse, JobEnvelope, ResourceClass, WorkerCapability
from zero_ttt_worker import ControlClient, JobContext, JobResult, WorkerRunner, run_worker
from zero_ttt_worker.client import ControlClientError


@pytest.fixture
def job():
    return JobEnvelope(
        job_id="job-test",
        workflow_id="workflow-test",
        kind="data.scan",
        capability=WorkerCapability.DATA,
        resource_class=ResourceClass.DATA_WRITER,
        attempt=1,
        lease_token="token",
        lease_expires_ns=10_000_000_000,
        idempotency_key="test",
        payload={},
    )


@pytest.fixture
def client(job):
    result = create_autospec(ControlClient, instance=True)
    result.lease.return_value = job
    result.heartbeat.return_value = HeartbeatResponse(
        lease_expires_ns=20_000_000_000, cancel_requested=False
    )
    return result


def runner_for(client, handler):
    return WorkerRunner(
        client,
        worker_id="data",
        capability=WorkerCapability.DATA,
        version="test",
        handlers={"data.scan": handler},
        lease_seconds=10,
        idle_seconds=0,
    )


def assert_heartbeat_stopped(job):
    assert all(thread.name != f"heartbeat-{job.job_id[:8]}" for thread in threading.enumerate())


def test_run_once_does_not_retry_registration_failure(client):
    client.register.side_effect = ControlClientError("offline")
    runner = runner_for(client, lambda *_: JobResult())
    with pytest.raises(ControlClientError, match="offline"):
        runner.run_once()
    client.register.assert_called_once()
    client.lease.assert_not_called()


def test_continuous_runner_recovers_from_registration_failure(client, job, caplog):
    client.register.side_effect = [ControlClientError("offline"), None]

    def execute(*_args):
        runner.request_stop()
        return JobResult({"done": True})

    runner = runner_for(client, execute)
    runner.run_forever()
    assert client.register.call_count == 2
    client.lease.assert_called_once()
    assert client.complete.call_args.args[1].result == {"done": True}
    assert "offline" in caplog.text
    assert_heartbeat_stopped(job)


def test_stop_during_registration_does_not_start_polling(client):
    runner = runner_for(client, lambda *_: JobResult())
    client.register.side_effect = lambda *_: runner.request_stop()
    runner.run_forever()
    client.lease.assert_not_called()


@pytest.mark.parametrize("error_type", [ValueError, TypeError, RuntimeError])
def test_failed_event_does_not_prevent_failure_report(client, job, error_type, caplog):
    def execute(*_args):
        raise error_type("handler failed")

    def event(_job, _worker, message):
        if message.kind == "job.failed":
            raise ControlClientError("events unavailable")
        return 1

    client.event.side_effect = event
    runner_for(client, execute).run_once()
    client.fail.assert_called_once()
    request = client.fail.call_args.args[1]
    assert request.retryable == (error_type is RuntimeError)
    assert request.error_type == error_type.__name__
    assert "events unavailable" in caplog.text
    client.complete.assert_not_called()
    assert_heartbeat_stopped(job)


def test_failure_report_network_error_is_logged_and_cleans_heartbeat(client, job, caplog):
    runner = runner_for(client, lambda *_: JobResult())
    runner.handlers.clear()
    client.fail.side_effect = ControlClientError("failure endpoint unavailable")
    assert runner.run_once()
    assert "failure endpoint unavailable" in caplog.text
    assert "no handler" in client.fail.call_args.args[1].message
    assert_heartbeat_stopped(job)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_process_interrupt_propagates_after_heartbeat_cleanup(client, job, error_type):
    def execute(*_args):
        raise error_type()

    with pytest.raises(error_type):
        runner_for(client, execute).run_once()
    client.fail.assert_not_called()
    client.complete.assert_not_called()
    assert_heartbeat_stopped(job)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_shutdown_drains_current_job_and_keeps_renewing(client, job, signum):
    previous = signal.getsignal(signum)
    renewed = threading.Event()

    def heartbeat(*_args):
        renewed.set()
        return HeartbeatResponse(lease_expires_ns=20_000_000_000, cancel_requested=False)

    def execute(_job, context):
        signal.raise_signal(signum)
        assert renewed.wait(timeout=5), "graceful shutdown stopped lease renewal"
        assert not context.cancel_requested
        return JobResult({"drained": True})

    client.heartbeat.side_effect = heartbeat
    run_worker(runner_for(client, execute))
    assert client.complete.call_args.args[1].result == {"drained": True}
    client.lease.assert_called_once()
    client.fail.assert_not_called()
    assert signal.getsignal(signum) == previous
    assert_heartbeat_stopped(job)


@pytest.mark.parametrize("leased", [True, False])
def test_stop_during_inflight_poll_finishes_any_acquired_job(client, job, leased):
    def poll(_request):
        signal.raise_signal(signal.SIGTERM)
        return job if leased else None

    client.lease.side_effect = poll
    run_worker(runner_for(client, lambda *_: JobResult()))
    client.lease.assert_called_once()
    assert client.complete.call_count == int(leased)
    assert_heartbeat_stopped(job)


def test_signal_handlers_are_restored_on_process_exception(client):
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    client.register.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_worker(runner_for(client, lambda *_: JobResult()))
    assert {signum: signal.getsignal(signum) for signum in previous} == previous


@pytest.mark.parametrize("lost_connection", [True, False])
def test_renewal_loss_or_user_cancel_requests_cooperative_stop(client, job, lost_connection):
    if lost_connection:
        client.heartbeat.side_effect = ControlClientError("lease lost")
    else:
        client.heartbeat.return_value = HeartbeatResponse(
            lease_expires_ns=20_000_000_000, cancel_requested=True
        )
    stop = create_autospec(threading.Event, instance=True)
    stop.wait.side_effect = [False, True]
    context = JobContext(client, "data", job)
    runner_for(client, lambda *_: JobResult())._heartbeat(job, context, stop)
    assert context.cancel_requested
    request = client.heartbeat.call_args.args[1]
    assert request.worker_id == "data" and request.lease_token == job.lease_token
