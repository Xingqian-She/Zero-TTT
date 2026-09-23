"""Container-side evidence and isolation checks for the S01 acceptance runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

E2E_SERVICES = ("control", "data-worker", "trainer-worker", "selfplay-worker", "ui")
GPU_SERVICES = {"gpu-smoke", "trainer-worker", "selfplay-worker"}


def workspace_origins(root: Path) -> dict[str, str]:
    workspace = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    origins = {}
    for pattern in workspace["tool"]["uv"]["workspace"]["members"]:
        for member in sorted(root.glob(pattern)):
            source = member / "src"
            for path in sorted(source.rglob("*.py")):
                parts = path.relative_to(source).with_suffix("").parts
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                module = ".".join(parts)
                spec = importlib.util.find_spec(module)
                if spec is None or spec.origin is None:
                    raise ValueError(f"missing workspace module: {module}")
                if Path(spec.origin).resolve() != path.resolve():
                    raise ValueError(
                        f"wrong import origin: {module}: {spec.origin}; expected {path}"
                    )
                origins[module] = str(Path(spec.origin).resolve())
    if not origins:
        raise ValueError("no workspace modules checked")
    return origins


def environment(root: Path) -> dict[str, Any]:
    import torch
    from zero_ttt.config import load_config

    _require(not torch.cuda.is_available(), "CPU test container unexpectedly has CUDA access")
    origins = workspace_origins(root)
    paths = set()
    for directory in (
        "packages",
        "services",
        "tools/admin-cli",
        "scripts",
        "configs",
        "tests",
        "docs",
        "docker",
    ):
        for path in (root / directory).rglob("*"):
            if path.is_file() and (
                path.suffix in {".py", ".ps1", ".toml", ".md", ".lock", ".cfg"}
                or path.name.startswith("Dockerfile")
            ):
                paths.add(path)
    paths.update(
        root / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "requirements.lock",
            "compose.yaml",
            "compose.e2e.yaml",
            ".dockerignore",
            "README.md",
        )
    )
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    config = load_config(root / "configs/acceptance/s01.toml")
    return {
        "python": sys.version,
        "test_cuda_available": False,
        "platform": platform.platform(),
        "origins": origins,
        "dependencies": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "files": hashes,
        "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "config_sha256": config.sha256,
        "effective_config": json.loads(config.canonical_json()),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _host_path(value: str) -> str:
    return value.replace("\\", "/").rstrip("/").casefold()


def _check_mounts(
    mounts: list[dict[str, Any]], service: str, workspace: str, volumes: set[str], *, runtime: bool
) -> None:
    targets = set()
    for mount in mounts:
        kind = mount.get("Type" if runtime else "type")
        target = mount.get("Destination" if runtime else "target")
        source = mount.get("Source" if runtime else "source", "")
        readonly = not mount.get("RW", True) if runtime else mount.get("read_only", False)
        targets.add(target)
        if kind == "tmpfs":
            _require(target == "/tmp", f"{service}: unexpected tmpfs {target}")
        elif kind == "bind":
            expected_target = "/profiles" if service == "control" else "/workspace"
            expected_source = (
                workspace + "/configs/acceptance" if service == "control" else workspace
            )
            _require(
                service in {"test", "gpu-smoke", "e2e-driver", "control"},
                f"{service}: business bind mount",
            )
            _require(
                target == expected_target and readonly,
                f"{service}: unsafe bind target or writable source",
            )
            # Docker Desktop translates host drive paths in container inspect output.
            normalized = _host_path(source)
            expected = _host_path(expected_source)
            translated = "/run/desktop/mnt/host/" + expected.replace(":", "")
            _require(
                normalized in {expected, translated}, f"{service}: unexpected bind source {source}"
            )
        elif kind == "volume":
            name = mount.get("Name") if runtime else source
            _require(
                service not in {"test", "gpu-smoke"} and name in volumes,
                f"{service}: foreign data volume {name}",
            )
            if target == "/raw":
                _require(readonly, "raw data must be read-only")
        else:
            raise ValueError(f"{service}: unexpected mount type {kind}")
    required = "/profiles" if service == "control" else "/workspace"
    if service in {"test", "gpu-smoke", "e2e-driver", "control"}:
        _require(required in targets, f"{service}: required bind missing")


def check_compose(payload: dict[str, Any]) -> None:
    workspace, base, e2e = payload["workspace"], payload["base"], payload["e2e"]
    project = payload["project"]
    _require(project.startswith("zero-ttt-s01-"), "E2E project must be unique to S01")
    for name in ("test", "gpu-smoke"):
        service = base["services"][name]
        _require(service.get("network_mode") == "none", f"{name}: network must be disabled")
        _require(
            bool(service.get("gpus")) == (name in GPU_SERVICES), f"{name}: GPU reservation mismatch"
        )
        _require(
            not service.get("devices") and not service.get("privileged"),
            f"{name}: unexpected device access",
        )
        _require(not service.get("ports"), f"{name}: published ports")
        _check_mounts(service.get("volumes", []), name, workspace, set(), runtime=False)
    volumes = e2e["volumes"]
    for volume in volumes.values():
        _require(
            not volume.get("external") and volume["name"].startswith(project + "_"),
            "E2E volume is not project-owned",
        )
    for name in (*E2E_SERVICES, "e2e-driver", "tensorboard"):
        service = e2e["services"][name]
        _require(not service.get("ports"), f"{name}: published E2E port")
        _require(not service.get("network_mode"), f"{name}: unexpected network mode")
        _require(
            bool(service.get("gpus")) == (name in GPU_SERVICES), f"{name}: GPU reservation mismatch"
        )
        _require(
            not service.get("devices") and not service.get("privileged"),
            f"{name}: unexpected device access",
        )
        _check_mounts(service.get("volumes", []), name, workspace, set(volumes), runtime=False)


def check_runtime(payload: dict[str, Any]) -> None:
    workspace, project = payload["workspace"], payload["project"]
    volumes = {item["name"] for item in payload["e2e"]["volumes"].values()}
    _require(bool(payload["containers"]), "no containers inspected")
    for container in payload["containers"]:
        labels = container["Config"]["Labels"]
        service = labels["com.docker.compose.service"]
        _require(
            service in {*E2E_SERVICES, "test", "gpu-smoke", "e2e-driver"},
            "unexpected inspected service",
        )
        host = container["HostConfig"]
        requests = host.get("DeviceRequests") or []
        has_gpu = any(
            "gpu" in group for request in requests for group in request.get("Capabilities", [])
        )
        _require(
            has_gpu == (service in GPU_SERVICES), f"{service}: actual GPU reservation mismatch"
        )
        _require(
            not host.get("Privileged") and not host.get("Devices"),
            f"{service}: unsafe device access",
        )
        _require(not host.get("PortBindings"), f"{service}: actual published ports")
        if service in {"test", "gpu-smoke"}:
            _require(host["NetworkMode"] == "none", f"{service}: actual network must be disabled")
        else:
            _require(labels["com.docker.compose.project"] == project, "foreign E2E container")
            _require(host["NetworkMode"] == project + "_default", "foreign E2E network")
        _check_mounts(container["Mounts"], service, workspace, volumes, runtime=True)


def check_report(report: dict[str, Any]) -> None:
    stages = report["stages"]
    passed = all(stages.get(name) == "passed" for name in ("preflight", "cpu", "gpu", "e2e"))
    _require(report["s01_passed"] == passed, "overall S01 status disagrees with executed stages")
    for command in report["commands"]:
        if command["exit_code"] != 0:
            _require(not report["s01_passed"], "failed command cannot yield S01 success")
            _require(
                stages.get(command["stage"]) != "passed",
                "failed command cannot yield stage success",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("environment", "compose", "runtime", "report", "hold"))
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.mode == "hold":
        time.sleep(3600)
    elif args.mode == "environment":
        print(json.dumps(environment(Path("/workspace")), indent=2))
    else:
        if args.input is None:
            parser.error("--input is required")
        payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
        {"compose": check_compose, "runtime": check_runtime, "report": check_report}[args.mode](
            payload
        )
        print(f"S01 {args.mode} check passed.")


if __name__ == "__main__":
    main()
