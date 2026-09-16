"""Eval-level knobs for durable sandbox dumps (`[archive]`)."""

from pydantic import Field
from pydantic_config import BaseConfig


class ArchiveConfig(BaseConfig):
    """Operator policy for host-side archival. The default dump is `/logs/artifacts`
    plus `task.data.artifacts`; `extra` adds to that set."""

    extra: list[str] = Field(default_factory=list)
    """Additional sandbox paths to dump (relative = runtime workdir)."""
    max_mb: int = Field(256, ge=1)
    """Ceiling for one rollout's host dump, in MiB. Collection still buffers tars
    in RAM, so this also bounds memory. Independent of grading's 32MB cap."""
