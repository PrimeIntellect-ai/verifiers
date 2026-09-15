"""Eval-level knobs for durable sandbox dumps (`[archive]`)."""

from pydantic import Field
from pydantic_config import BaseConfig


class ArchiveConfig(BaseConfig):
    """Operator policy for host-side archival. The default dump is `/logs/artifacts`
    plus `task.data.artifacts`; these knobs only add or filter that set."""

    extra: list[str] = Field(default_factory=list)
    """Additional sandbox paths to dump (relative = runtime workdir)."""
    exclude: list[str] = Field(default_factory=list)
    """`tar --exclude` patterns applied to every archived root, including the
    convention dir and task grading artifacts."""
