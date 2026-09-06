"""Main-thread entrypoint: drain the current finite job on process shutdown."""

from __future__ import annotations

import logging
import signal
from types import FrameType

from zero_ttt_worker.runner import WorkerRunner


def run_worker(runner: WorkerRunner) -> None:
    logging.basicConfig(level=logging.INFO)

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        logging.getLogger(__name__).info(
            "Worker %s received signal %s; finishing its current job", runner.worker_id, signum
        )
        runner.request_stop()

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        for signum in previous:
            signal.signal(signum, request_stop)
        runner.run_forever()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
