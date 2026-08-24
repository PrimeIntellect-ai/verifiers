"""Environment-scaffold CLI configuration."""

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class InitConfig(BaseConfig):
    name: str = Field(min_length=1)
    """The new environment id, e.g. `my-task-v1` (positional: `init my-task-v1`)."""
    path: str = Field("./environments", validation_alias=AliasChoices("path", "p"))
    """Parent directory the package is created in (default `./environments`)."""
    add_tool: bool = Field(False, validation_alias=AliasChoices("add_tool", "T"))
    """Also scaffold a `vf.Toolset` declared on the task (`-T`)."""
    add_harness: bool = Field(False, validation_alias=AliasChoices("add_harness", "H"))
    """Also scaffold a custom `vf.Harness` (`harness.py`), selectable via `--env.agent.harness.id <name>` (`-H`)."""
    force: bool = False
    """Overwrite an existing environment package (default: refuse if it already exists)."""
