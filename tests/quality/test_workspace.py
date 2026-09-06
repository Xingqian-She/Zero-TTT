"""Verify that Docker checks use this checkout and its locked development tools."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_all_workspace_modules_resolve_to_current_sources() -> None:
    root = Path.cwd()
    workspace = tomllib.loads((root / "pyproject.toml").read_text())
    checked = 0
    for pattern in workspace["tool"]["uv"]["workspace"]["members"]:
        for member in root.glob(pattern):
            source = member / "src"
            for path in source.rglob("*.py"):
                parts = path.relative_to(source).with_suffix("").parts
                if parts[-1] == "__init__":
                    parts = parts[:-1]
                module = ".".join(parts)
                spec = importlib.util.find_spec(module)
                assert spec is not None and spec.origin is not None, module
                assert Path(spec.origin).resolve() == path.resolve(), (module, spec.origin)
                checked += 1
    assert checked > 0


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
