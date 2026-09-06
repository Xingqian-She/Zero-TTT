"""Data worker entrypoint."""

from __future__ import annotations

from importlib.metadata import version

from zero_ttt_contracts import WorkerCapability
from zero_ttt_worker import ControlClient, WorkerRunner, run_worker

from zero_ttt_data.handlers import DataJobHandlers
from zero_ttt_data.settings import DataSettings


def main() -> None:
    settings = DataSettings.from_environment()
    runner = WorkerRunner(
        ControlClient(settings.control_url),
        worker_id=settings.worker_id,
        capability=WorkerCapability.DATA,
        version=version("zero-ttt-data-service"),
        handlers=DataJobHandlers(settings).mapping(),
    )
    run_worker(runner)


if __name__ == "__main__":
    main()
