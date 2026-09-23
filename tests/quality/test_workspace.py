"""Verify that Docker checks use this checkout and its locked development tools."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

from scripts.check_s01 import workspace_origins


def test_all_workspace_modules_resolve_to_current_sources() -> None:
    assert workspace_origins(Path.cwd())


def test_development_constraints_match_uv_lock() -> None:
    workspace = tomllib.loads(Path("pyproject.toml").read_text())
    lock = tomllib.loads(Path("uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    pending = [Requirement(value).name for value in workspace["dependency-groups"]["dev"]]
    expected: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name in expected:
            continue
        package = packages[name]
        expected[name] = f"=={package['version']}"
        pending.extend(dependency["name"] for dependency in package.get("dependencies", ()))
    requirements = (
        Requirement(line)
        for line in Path("docker/services/dev.requirements.lock").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert {
        requirement.name: str(requirement.specifier) for requirement in requirements
    } == expected
