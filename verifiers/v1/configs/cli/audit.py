"""Post-run trajectory analysis and task-source rubric reviews."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.judge import JudgeConfig


class AuditConfig(BaseConfig):
    action: Literal["analyze", "check"]
    """Analyze saved traces with a judge, or check task source."""
    path: Path
    """Saved run directory; for `check`, the task source directory."""
    output: Path | None = None
    """New JSON sidecar. Defaults to a timestamped file in `<path>/audit/`."""
    rubric: Path
    """Explicit JSON/TOML rubric for `analyze` and `check`, using the native criteria format."""
    source: Path | None = None
    """Optional task source directory to include when analyzing saved traces."""
    judge: JudgeConfig = JudgeConfig()
    """Model endpoint and sampling for the review."""
    max_concurrent: int = Field(4, ge=1)
    """Maximum concurrent trace reviews."""

    @model_validator(mode="after")
    def validate_action(self):
        if self.action == "check" and self.source is not None:
            raise ValueError("check reads its source from --path")
        return self
