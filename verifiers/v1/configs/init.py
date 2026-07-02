"""The `InitConfig`: the config the `init` CLI parses.

`init` scaffolds a new environment package (see `verifiers.v1.cli.init`); this config is just
the "what to scaffold" knobs — the new env name, where to create it, and which optional pieces
(tool server, user simulator, custom harness) to include.
"""

from pydantic import AliasChoices, Field
from pydantic_config import BaseConfig


class InitConfig(BaseConfig):
    """What to scaffold. The `name` is the new environment id (a leading bare token, e.g.
    `init my-task-v1`); the package dir, ids, and class names are derived from it. The
    `--add-*` flags add optional pieces (tool server, user simulator, custom harness)."""

    name: str = ""
    """The new environment id, e.g. `my-task-v1` (positional: `init my-task-v1`)."""
    path: str = Field("./environments", validation_alias=AliasChoices("path", "p"))
    """Parent directory the package is created in (default `./environments`)."""
    add_tool: bool = Field(False, validation_alias=AliasChoices("add_tool", "T"))
    """Also scaffold a `vf.Toolset` tool server (`servers/tool.py`), wired into the taskset (`-T`)."""
    add_user: bool = Field(False, validation_alias=AliasChoices("add_user", "U"))
    """Also scaffold a `vf.User` simulator (`servers/user.py`), wired into the taskset (`-U`)."""
    add_harness: bool = Field(False, validation_alias=AliasChoices("add_harness", "H"))
    """Also scaffold a custom `vf.Harness` (`harness.py`), selectable via `--solver.harness.id <name>` (`-H`)."""
    v0: bool = False
    """Scaffold a legacy v0 environment (a `load_environment` package) instead of a v1 taskset."""
    force: bool = False
    """Overwrite an existing environment package (default: refuse if it already exists)."""
