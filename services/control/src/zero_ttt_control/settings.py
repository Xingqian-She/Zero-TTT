"""Environment-only runtime settings for the control service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ControlSettings:
    database_path: Path
    profile_root: Path
    host: str
    port: int

    @classmethod
    def from_environment(cls) -> ControlSettings:
        return cls(
            database_path=Path(
                os.environ.get("ZERO_TTT_CONTROL_DB", "/state/control/control.sqlite")
            ),
            profile_root=Path(os.environ.get("ZERO_TTT_PROFILE_ROOT", "/profiles")),
            host=os.environ.get("ZERO_TTT_CONTROL_HOST", "0.0.0.0"),
            port=int(os.environ.get("ZERO_TTT_CONTROL_PORT", "8090")),
        )
