"""Self-play worker entrypoint."""

from __future__ import annotations

from importlib.metadata import version

from zero_ttt_contracts import WorkerCapability
from zero_ttt_worker import ControlClient, WorkerRunner, run_worker

from zero_ttt_selfplay_worker.jobs import SelfPlayJobHandler
from zero_ttt_selfplay_worker.settings import SelfPlaySettings


def main() -> None:
    settings = SelfPlaySettings.from_environment()
    runner = WorkerRunner(
        ControlClient(settings.control_url),
        worker_id=settings.worker_id,
        capability=WorkerCapability.SELFPLAY,
        version=version("zero-ttt-selfplay-service"),
        handlers=SelfPlayJobHandler(settings).mapping(),
        lease_seconds=120,
    )
    run_worker(runner)


if __name__ == "__main__":
    main()
