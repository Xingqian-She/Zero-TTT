"""Run one complete concurrent OpenSpiel self-play round on CUDA."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from zero_ttt.config import load_config
from zero_ttt.model import PolicyValueTransformer
from zero_ttt.precision import configure_strict_fp32
from zero_ttt_dataset import ShardStore
from zero_ttt_selfplay_worker.service import SelfPlayService
from zero_ttt_trainer.checkpoint import CheckpointManager, checkpoint_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acceptance/s01.toml")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the self-play GPU smoke test")
    configure_strict_fp32()
    config = load_config(arguments.config)
    torch.manual_seed(config.seed)
    if config.runtime.device != "cuda" or config.game.max_moves != 2:
        raise ValueError("self-play smoke requires CUDA and a two-move budget")

    with tempfile.TemporaryDirectory(prefix="zero-ttt-selfplay-gpu-") as temporary:
        root = Path(temporary)
        model = PolicyValueTransformer(config.model)
        manager = CheckpointManager(root / "model", keep=1)
        publication = manager.save_publication(
            "gpu-smoke",
            1,
            1,
            model.state_dict(),
            checkpoint_metadata(config.canonical_json(), config.sha256),
        )
        shard_root = root / "selfplay"
        with SelfPlayService(config, publication, store_root=shard_root) as service:
            summary = service.collect(games=4, seed=11, target_shard_bytes=1024 * 1024)
            gpu_peak = service.gpu_peak_allocated_bytes()

        records = tuple(
            record
            for path in sorted(ShardStore(shard_root, read_only=True).trajectory_dir.glob("*.npz"))
            for record in ShardStore(shard_root, read_only=True).read_trajectories(path)
        )
        assert summary.requested_games == 4
        assert summary.collected_games == 4
        assert summary.skipped_games == 0
        assert len(records) == 4
        assert len({record.game_id for record in records}) == 4
        assert all(record.trainable_position_count == 2 for record in records)
        assert summary.batching.requests > 0
        assert summary.batching.batches > 0
        assert gpu_peak > 0
        print(
            json.dumps(
                {
                    **asdict(summary),
                    "gpu_peak_allocated_bytes": gpu_peak,
                    "config_sha256": config.sha256,
                    "effective_config": json.loads(config.canonical_json()),
                },
                indent=2,
            )
        )
        print("Concurrent CUDA self-play smoke test passed.")


if __name__ == "__main__":
    main()
