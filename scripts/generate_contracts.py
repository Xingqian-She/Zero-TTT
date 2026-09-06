"""Generate committed OpenAPI and JSON Schema documents from source models."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from zero_ttt_contracts import ArtifactRef, DomainEvent, JobEnvelope, RunSpec
from zero_ttt_control.api import create_app
from zero_ttt_control.store import ControlStore

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "generated" / "contracts"


def documents() -> dict[Path, dict[str, Any]]:
    with tempfile.TemporaryDirectory() as directory:
        store = ControlStore(Path(directory) / "control.sqlite")
        try:
            openapi = create_app(store, profile_root=ROOT / "configs" / "profiles").openapi()
        finally:
            store.close()
    models = (ArtifactRef, DomainEvent, JobEnvelope, RunSpec)
    result = {Path("openapi.json"): openapi}
    result.update(
        {Path("schemas") / f"{model.__name__}.json": model.model_json_schema() for model in models}
    )
    return result


def _encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def generate(*, check: bool) -> list[str]:
    errors: list[str] = []
    expected = documents()
    for relative, value in expected.items():
        destination = OUTPUT / relative
        payload = _encoded(value)
        if check:
            if not destination.is_file() or destination.read_text(encoding="utf-8") != payload:
                errors.append(f"generated contract is stale: {destination.relative_to(ROOT)}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8", newline="\n")
    if check:
        expected_paths = {OUTPUT / relative for relative in expected}
        actual_paths = set(OUTPUT.rglob("*.json")) if OUTPUT.exists() else set()
        for unexpected in sorted(actual_paths - expected_paths):
            errors.append(f"unexpected generated contract: {unexpected.relative_to(ROOT)}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    errors = generate(check=arguments.check)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Contract documents are current." if arguments.check else f"Wrote {OUTPUT}.")


if __name__ == "__main__":
    main()
