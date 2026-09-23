"""Prepare and drive the isolated Compose service-flow acceptance test."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import httpx
from zero_ttt.config import config_from_mapping, load_config
from zero_ttt_contracts import ArtifactRef
from zero_ttt_dataset import LocalArtifactStore
from zero_ttt_dataset.records import stable_game_id

RUN_NAME = "compose-e2e"
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
BOOTSTRAP_PARAMETERS = {"trial_games": 1, "validation_fraction": 0.25, "seed": 7}


def prepare(root: Path) -> None:
    root = root.resolve()
    isolated_directories = (
        "raw",
        "work",
        "artifacts/data",
        "artifacts/models",
        "artifacts/selfplay",
        "state/control",
        "state/data",
    )
    for relative in isolated_directories:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty E2E directory: {directory}")
    (root / "raw/katago/g170/selfplay").mkdir(parents=True)
    valid_sgf = (
        b"(;FF[4]GM[1]SZ[19]HA[0]KM[0]"
        b"RU[koPOSITIONALscoreAREAtaxNONEsui1]RE[0]"
        b"C[startTurnIdx=1,mode=normal];B[aa];W[bb];B[];W[])"
    )
    archive = root / "raw/katago/g170/selfplay/e2e.zip"
    member_path = "net/sgfs/games.sgfs"
    member = zipfile.ZipInfo(member_path, date_time=(2020, 1, 1, 0, 0, 0))
    member.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(member, b"\n".join((valid_sgf,) * 32))
    asset_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    validation_games = sum(
        int.from_bytes(
            hashlib.sha256(
                f"7:{stable_game_id('katago-g170', asset_sha256, member_path, ordinal)}".encode(
                    "ascii"
                )
            ).digest()[:8],
            "big",
        )
        < int(0.25 * (1 << 64))
        for ordinal in range(32)
    )
    if not 0 < validation_games < 32:
        raise RuntimeError("deterministic E2E corpus does not populate both data splits")
    print(
        f"Prepared isolated E2E root: {root} "
        f"(train={32 - validation_games}, validation={validation_games})"
    )


def _post(client: httpx.Client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=payload)
    response.raise_for_status()
    return response.json()


def _wait_workflow(
    client: httpx.Client,
    workflow_id: str,
    *,
    timeout_seconds: float = 240.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_state = ""
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/workflows/{workflow_id}")
        response.raise_for_status()
        workflow = response.json()
        state = str(workflow["state"])
        if state != last_state:
            print(f"workflow {workflow_id}: {state}")
            last_state = state
        if state in TERMINAL_STATES:
            jobs_response = client.get("/api/v1/jobs", params={"workflow_id": workflow_id})
            jobs_response.raise_for_status()
            jobs = jobs_response.json()["jobs"]
            if state != "succeeded":
                raise RuntimeError(json.dumps(jobs, indent=2, ensure_ascii=False))
            if not jobs or any(job["state"] != "succeeded" for job in jobs):
                raise RuntimeError("workflow succeeded without all jobs succeeding")
            return {"workflow": workflow, "jobs": jobs}
        time.sleep(0.25)
    raise TimeoutError(f"workflow {workflow_id} did not finish in {timeout_seconds}s")


def _submit(
    client: httpx.Client,
    template: str,
    *,
    run_id: str = "",
    parameters: dict[str, Any] | None = None,
) -> str:
    body = {
        "run_id": run_id,
        "parameters": parameters or {},
        "idempotency_key": f"compose-e2e-{template}-v1",
    }
    return str(_post(client, f"/api/v1/workflows/{template}", body)["workflow_id"])


def _require_empty_control(client: httpx.Client) -> None:
    initial = client.get("/api/v1/snapshot")
    initial.raise_for_status()
    if any(initial.json().get(name) for name in ("workflows", "jobs", "runs", "artifacts")):
        raise RuntimeError("isolated Control state is not empty")


def recovery_start(client: httpx.Client, state_path: Path) -> dict[str, Any]:
    """Act as a worker that disappears while holding the first bootstrap lease."""
    _require_empty_control(client)
    workflow_id = _submit(client, "data-bootstrap", parameters=BOOTSTRAP_PARAMETERS)
    registration = {"worker_id": "e2e-lost-worker", "capability": "data", "version": "test"}
    _post(client, "/internal/v1/workers/register", registration)
    job = _post(
        client,
        "/internal/v1/jobs/lease",
        {
            "worker_id": registration["worker_id"],
            "capability": "data",
            "lease_seconds": 60,
            "wait_seconds": 0,
        },
    )["job"]
    if job is None or job["workflow_id"] != workflow_id or job["attempt"] != 1:
        raise RuntimeError("recovery probe did not acquire the first bootstrap attempt")
    response = client.post(
        f"/internal/v1/jobs/{job['job_id']}/events",
        json={
            "event_id": "e2e-before-restart",
            "job_id": job["job_id"],
            "kind": "e2e.before-restart",
        },
        headers={"X-Worker-ID": registration["worker_id"], "X-Lease-Token": job["lease_token"]},
    )
    response.raise_for_status()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "job": job,
                "worker_id": registration["worker_id"],
                "sequence": response.json()["sequence"],
            }
        ),
        encoding="utf-8",
    )
    print(f"Worker disappeared holding job {job['job_id']}; restart Control and UI now.")
    return {"workflow_id": workflow_id, "job_id": job["job_id"], "attempt": job["attempt"]}


def recovery_check(client: httpx.Client, state_path: Path) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job = state["job"]
    restored = client.get(f"/api/v1/jobs/{job['job_id']}").raise_for_status().json()
    if restored["state"] != "leased" or restored["attempt"] != 1:
        raise RuntimeError("Control restart did not preserve the active lease")
    if time.time_ns() >= job["lease_expires_ns"]:
        raise RuntimeError("restart check must begin before the probe lease expires")
    available = _post(
        client,
        "/internal/v1/jobs/lease",
        {
            "worker_id": "e2e-contender",
            "capability": "data",
            "wait_seconds": 0,
        },
    )["job"]
    if available is not None:
        raise RuntimeError("a live resource lease was reassigned after restart")
    events = (
        client.get("/api/v1/events", params={"after": state["sequence"] - 1})
        .raise_for_status()
        .json()["events"]
    )
    if not events or events[0]["event_id"] != "e2e-before-restart":
        raise RuntimeError("event cursor was not persisted across restart")
    while time.time_ns() <= job["lease_expires_ns"]:
        time.sleep(0.25)
    response = client.post(
        f"/internal/v1/jobs/{job['job_id']}/heartbeat",
        json={
            "worker_id": state["worker_id"],
            "lease_token": job["lease_token"],
        },
    )
    if response.status_code != 409:
        raise RuntimeError("an expired worker lease was allowed to renew")
    print("Active lease and event cursor survived restart; expired token rejected.")
    return {
        "active_lease_persisted": True,
        "event_cursor_persisted": True,
        "expired_token_status": 409,
    }


def verify_training_job(
    job: dict[str, Any], run_id: str, optimizer_step: int, effective_batch: int
) -> None:
    result = job["result"]
    if (
        job["run_id"] != run_id
        or result.get("run_id") != run_id
        or result.get("steps_executed") != 1
        or result.get("optimizer_step") != optimizer_step
        or result.get("samples_seen") != optimizer_step * effective_batch
    ):
        raise RuntimeError(f"training did not execute the requested step for this run: {job}")


def verify_artifacts(client: httpx.Client, expected: dict[str, str]) -> list[dict[str, Any]]:
    artifacts = client.get("/api/v1/artifacts").raise_for_status().json()["artifacts"]
    by_id = {item["artifact_id"]: item for item in artifacts}
    selected = []
    store = LocalArtifactStore("/datasets/artifacts")
    for artifact_id, kind in expected.items():
        item = by_id.get(artifact_id)
        if item is None or item["kind"] != kind:
            raise RuntimeError(f"missing artifact for this workflow: {artifact_id} ({kind})")
        store.verify(ArtifactRef.model_validate(item))
        selected.append(item)
    return selected


def verify_profile(run: dict[str, Any], config_path: str) -> int:
    expected = load_config(config_path)
    actual = config_from_mapping(run["profile"])
    if actual.sha256 != expected.sha256 or actual.runtime.device != "cuda":
        raise RuntimeError("E2E Run did not freeze the requested CUDA configuration")
    return actual.training.effective_batch_size


def bootstrap(
    client: httpx.Client, state_path: Path, profile_id: str, config_path: str
) -> dict[str, Any]:
    recovery = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if recovery is None:
        _require_empty_control(client)
    data_workflow = _submit(
        client,
        "data-bootstrap",
        parameters=BOOTSTRAP_PARAMETERS,
    )
    data_result = _wait_workflow(client, data_workflow)
    if len(data_result["jobs"]) != 7:
        raise RuntimeError("data-bootstrap must complete all seven jobs")
    if recovery is not None:
        first = data_result["jobs"][0]
        if (
            data_workflow != recovery["workflow_id"]
            or first["job_id"] != recovery["job"]["job_id"]
            or first["attempt"] != 2
        ):
            raise RuntimeError("the lost job was not recovered exactly once")
        response = client.post(
            f"/internal/v1/jobs/{first['job_id']}/heartbeat",
            json={
                "worker_id": recovery["worker_id"],
                "lease_token": recovery["job"]["lease_token"],
            },
        )
        if response.status_code != 409:
            raise RuntimeError("the old token was accepted after job recovery")
    snapshot_ids = {
        job["kind"].rsplit("-", 1)[-1]: f"dataset.{job['result']['snapshot_id']}"
        for job in data_result["jobs"]
        if job["kind"].startswith("data.snapshot-")
    }
    datasets = verify_artifacts(
        client, {value: "dataset-snapshot" for value in snapshot_ids.values()}
    )
    if set(snapshot_ids) != {"train", "validation"} or any(
        int(item["labels"]["games"]) <= 0 for item in datasets
    ):
        raise RuntimeError("bootstrap must create nonempty train and validation snapshots")
    cold = next(
        item
        for item in datasets
        if item["labels"].get("split") == "train"
        and item["labels"].get("source_kind") == "external"
    )
    run = _post(
        client,
        "/api/v1/runs",
        {"name": RUN_NAME, "profile_id": profile_id, "cold_snapshot_id": cold["artifact_id"]},
    )
    effective_batch = verify_profile(run, config_path)
    cold_workflow = _submit(
        client,
        "cold-start",
        run_id=str(run["run_id"]),
        parameters={"steps": 1},
    )
    cold_result = _wait_workflow(client, cold_workflow)
    if len(cold_result["jobs"]) != 1:
        raise RuntimeError("cold-start must contain exactly one training job")
    verify_training_job(cold_result["jobs"][0], run["run_id"], 1, effective_batch)
    artifacts = verify_artifacts(
        client,
        {
            f"checkpoint.{run['run_id']}.1": "checkpoint",
            f"publication.{run['run_id']}.1": "publication",
        },
    )
    publications = client.get("/api/v1/publications").raise_for_status().json()["publications"]
    if len(publications) != 1:
        raise RuntimeError("cold-start did not publish exactly one model")
    events = client.get("/api/v1/events", params={"after": 0}).raise_for_status().json()["events"]
    sequences = [int(event["sequence"]) for event in events]
    if not sequences or sequences != sorted(set(sequences)):
        raise RuntimeError("persistent event sequence is empty, duplicated, or unordered")
    return {
        "data_workflow": data_result["workflow"]["state"],
        "cold_workflow": cold_result["workflow"]["state"],
        "run_id": run["run_id"],
        "cold_snapshot": cold["artifact_id"],
        "publication": publications[0]["artifact_id"],
        "last_event_sequence": sequences[-1],
        "jobs": data_result["jobs"] + cold_result["jobs"],
        "artifacts": datasets + artifacts,
        "profile_sha256": run["profile_sha256"],
        "config_sha256": load_config(config_path).sha256,
        "device": "cuda",
    }


def alpha(client: httpx.Client, config_path: str) -> dict[str, Any]:
    runs = client.get("/api/v1/runs").raise_for_status().json()["runs"]
    run = next(item for item in runs if item["name"] == RUN_NAME)
    effective_batch = verify_profile(run, config_path)
    before_events = (
        client.get("/api/v1/events", params={"after": 0}).raise_for_status().json()["events"]
    )
    cursor = int(before_events[-1]["sequence"])
    workflow_id = _submit(
        client,
        "alpha-zero-round",
        run_id=str(run["run_id"]),
        parameters={"games": 4, "steps": 1, "seed": 19},
    )
    result = _wait_workflow(client, workflow_id)
    jobs = {job["kind"]: job for job in result["jobs"]}
    if len(result["jobs"]) != 4 or set(jobs) != {
        "selfplay.collect",
        "data.admit-selfplay",
        "data.snapshot-selfplay",
        "trainer.mixture",
    }:
        raise RuntimeError("alpha-zero round must complete exactly its four jobs")
    verify_training_job(jobs["trainer.mixture"], run["run_id"], 2, effective_batch)
    collection = jobs["selfplay.collect"]["result"]
    if (
        collection["sealed_games"] != 4
        or collection["sealed_positions"] != 8
        or collection["gpu_peak_allocated_bytes"] <= 0
    ):
        raise RuntimeError("E2E self-play did not execute the four-game CUDA budget")
    artifacts = verify_artifacts(
        client,
        {
            f"selfplay.{workflow_id}": "selfplay-bundle",
            f"dataset.{jobs['data.snapshot-selfplay']['result']['snapshot_id']}": "dataset-snapshot",
            f"checkpoint.{run['run_id']}.2": "checkpoint",
            f"publication.{run['run_id']}.2": "publication",
        },
    )
    resumed = (
        client.get("/api/v1/events", params={"after": cursor}).raise_for_status().json()["events"]
    )
    sequences = [int(event["sequence"]) for event in resumed]
    if not sequences or min(sequences) <= cursor or sequences != sorted(set(sequences)):
        raise RuntimeError("event cursor did not resume strictly after the persisted checkpoint")
    return {
        "alpha_workflow": result["workflow"]["state"],
        "workflow_id": workflow_id,
        "jobs": result["jobs"],
        "artifacts": artifacts,
        "resumed_event_count": len(resumed),
        "last_event_sequence": sequences[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare_command = subcommands.add_parser("prepare")
    prepare_command.add_argument("root", type=Path)
    run_command = subcommands.add_parser("run")
    run_command.add_argument(
        "phase", choices=("recovery-start", "recovery-check", "bootstrap", "alpha")
    )
    run_command.add_argument("--url", default="http://control:8090")
    run_command.add_argument("--profile-id", default="s01")
    run_command.add_argument("--config", default="configs/acceptance/s01.toml")
    run_command.add_argument(
        "--recovery-state", type=Path, default=Path("/datasets/work/e2e-recovery.json")
    )
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepare(arguments.root)
        return
    with httpx.Client(base_url=arguments.url, timeout=30.0) as client:
        if arguments.phase == "alpha":
            result = alpha(client, arguments.config)
        elif arguments.phase == "bootstrap":
            result = bootstrap(
                client, arguments.recovery_state, arguments.profile_id, arguments.config
            )
        else:
            phases = {
                "recovery-start": recovery_start,
                "recovery-check": recovery_check,
            }
            result = phases[arguments.phase](client, arguments.recovery_state)
        output = json.dumps(result, indent=2)
        (arguments.recovery_state.parent / f"e2e-{arguments.phase}-result.json").write_text(
            output + "\n", encoding="utf-8"
        )
        print(output)


if __name__ == "__main__":
    main()
