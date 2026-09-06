"""Trainer worker entrypoint."""

from __future__ import annotations

from importlib.metadata import version

from zero_ttt_contracts import WorkerCapability
from zero_ttt_worker import ControlClient, WorkerRunner, run_worker

from zero_ttt_trainer.jobs import TrainingJobHandler
from zero_ttt_trainer.settings import TrainerSettings


def main() -> None:
    settings = TrainerSettings.from_environment()
    runner = WorkerRunner(
        ControlClient(settings.control_url),
        worker_id=settings.worker_id,
        capability=WorkerCapability.TRAINER,
        version=version("zero-ttt-trainer-service"),
        handlers=TrainingJobHandler(settings).mapping(),
        lease_seconds=120,
    )
    run_worker(runner)


if __name__ == "__main__":
    main()
