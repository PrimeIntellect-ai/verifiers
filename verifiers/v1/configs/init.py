"""Environment-scaffold CLI configuration."""

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class InitConfig(BaseConfig):
    name: str = ""
    """The new environment id, e.g. `my-task-v1` (positional: `init my-task-v1`)."""
    path: str = Field("./environments", validation_alias=AliasChoices("path", "p"))
    """Parent directory the package is created in (default `./environments`)."""
    add_tool: bool = Field(False, validation_alias=AliasChoices("add_tool", "T"))
    """Also scaffold a `vf.Toolset` declared on the task (`-T`)."""
    add_user: bool = Field(False, validation_alias=AliasChoices("add_user", "U"))
    """Also scaffold a `vf.User` declared on the task (`-U`)."""
    add_harness: bool = Field(False, validation_alias=AliasChoices("add_harness", "H"))
    """Also scaffold a custom `vf.Harness` (`harness.py`), selectable via `--env.agent.harness.id <name>` (`-H`)."""
    v0: bool = False
    """Scaffold a legacy v0 environment (a `load_environment` package) instead of a v1 taskset."""
    force: bool = False
    """Overwrite an existing environment package (default: refuse if it already exists)."""
