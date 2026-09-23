"""Fault cases which must never produce a credible acceptance pass."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from zero_ttt.config import load_config

from scripts.check_s01 import (
    E2E_SERVICES,
    GPU_SERVICES,
    check_compose,
    check_report,
    check_runtime,
    workspace_origins,
)
from scripts.compose_e2e_test import verify_training_job


def compose_fixture():
    def service(name):
        binds = []
        if name in {"test", "gpu-smoke", "e2e-driver", "control"}:
            binds = [
                {
                    "type": "bind",
                    "source": "/repo/configs/acceptance" if name == "control" else "/repo",
                    "target": "/profiles" if name == "control" else "/workspace",
                    "read_only": True,
                }
            ]
        return {"volumes": binds, "gpus": [{}] if name in GPU_SERVICES else []}

    base = {
        "services": {
            name: service(name) | {"network_mode": "none"} for name in ("test", "gpu-smoke")
        }
    }
    e2e = {
        "services": {name: service(name) for name in (*E2E_SERVICES, "e2e-driver", "tensorboard")},
        "volumes": {"data": {"name": "zero-ttt-s01-test_data"}},
    }
    return {"workspace": "/repo", "project": "zero-ttt-s01-test", "base": base, "e2e": e2e}


def test_valid_isolation():
    check_compose(compose_fixture())


@pytest.mark.parametrize(
    "change", ["write", "foreign-bind", "network", "gpu", "port", "foreign-volume", "driver-gpu"]
)
def test_unsafe_compose_rejected(change):
    payload = compose_fixture()
    service = payload["base"]["services"]["test"]
    if change == "write":
        service["volumes"][0]["read_only"] = False
    elif change == "foreign-bind":
        service["volumes"].append(
            {"type": "bind", "source": "/business", "target": "/datasets", "read_only": True}
        )
    elif change == "network":
        service["network_mode"] = "host"
    elif change == "gpu":
        service["gpus"] = [{}]
    elif change == "port":
        payload["e2e"]["services"]["control"]["ports"] = [8090]
    elif change == "foreign-volume":
        payload["e2e"]["volumes"]["data"]["name"] = "production_data"
    else:
        payload["e2e"]["services"]["e2e-driver"]["gpus"] = [{}]
    with pytest.raises(ValueError):
        check_compose(payload)


@pytest.mark.parametrize("change", [None, "write", "network", "gpu", "foreign-bind"])
def test_actual_container_checked(change):
    payload = compose_fixture()
    container = {
        "Config": {"Labels": {"com.docker.compose.service": "test"}},
        "HostConfig": {
            "NetworkMode": "none",
            "DeviceRequests": [],
            "Devices": [],
            "Privileged": False,
            "PortBindings": {},
        },
        "Mounts": [{"Type": "bind", "Source": "/repo", "Destination": "/workspace", "RW": False}],
    }
    payload["containers"] = [container]
    if change == "write":
        container["Mounts"][0]["RW"] = True
    elif change == "network":
        container["HostConfig"]["NetworkMode"] = "bridge"
    elif change == "gpu":
        container["HostConfig"]["DeviceRequests"] = [{"Capabilities": [["gpu"]]}]
    elif change == "foreign-bind":
        container["Mounts"][0]["Source"] = "/stale-checkout"
    if change is None:
        check_runtime(payload)
    else:
        with pytest.raises(ValueError):
            check_runtime(payload)


def test_wrong_import_origin_rejected(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers=["packages/*"]\n')
    source = tmp_path / "packages/example/src"
    source.mkdir(parents=True)
    (source / "example.py").write_text("")
    monkeypatch.setattr(
        "scripts.check_s01.importlib.util.find_spec",
        lambda _: SimpleNamespace(origin="/site-packages/example.py"),
    )
    with pytest.raises(ValueError, match="wrong import origin"):
        workspace_origins(tmp_path)


@pytest.mark.parametrize("state", ["not_run", "blocked", "failed", "running"])
def test_partial_report_cannot_pass(state):
    report = {
        "stages": {name: "passed" for name in ("preflight", "cpu", "gpu", "e2e")},
        "s01_passed": True,
        "commands": [],
    }
    report["stages"]["gpu"] = state
    with pytest.raises(ValueError):
        check_report(report)
    report["s01_passed"] = False
    check_report(report)


def test_failed_command_cannot_pass():
    report = {
        "stages": {name: "passed" for name in ("preflight", "cpu", "gpu", "e2e")},
        "s01_passed": True,
        "commands": [{"stage": "cpu", "exit_code": 1}],
    }
    with pytest.raises(ValueError, match="failed command"):
        check_report(report)


def test_training_success_requires_new_step_for_current_run():
    job = {
        "run_id": "new",
        "result": {"run_id": "new", "steps_executed": 1, "optimizer_step": 2, "samples_seen": 4},
    }
    verify_training_job(job, "new", 2, 2)
    for key, wrong_value in (
        ("run_id", "old"),
        ("steps_executed", 0),
        ("optimizer_step", 1),
        ("samples_seen", 2),
    ):
        invalid = copy.deepcopy(job)
        invalid["result"][key] = wrong_value
        with pytest.raises(RuntimeError):
            verify_training_job(invalid, "new", 2, 2)


def test_acceptance_budget_is_bounded_and_separate():
    config = load_config("configs/acceptance/s01.toml")
    assert (config.game.board_size, config.game.max_moves) == (19, 2)
    assert (config.training.batch_size, config.training.accumulation_steps) == (2, 1)
    assert config.search.max_simulations == 2
    assert (config.selfplay.actor_count, config.selfplay.inference_batch_size) == (4, 4)
    assert config.runtime.device == "cuda" and config.runtime.ema_device == "cpu"
    assert not config.execution.compile_model and not config.selfplay.compile_inference
    assert Path("configs/acceptance/s01.toml").parent != Path("configs/profiles")
