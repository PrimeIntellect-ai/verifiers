"""Prime-hosted command plugin contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Sequence

PRIME_PLUGIN_API_VERSION = 1


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def _resolve_workspace_python(cwd: Path | None = None) -> str:
    uv_project_env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if uv_project_env:
        candidate = _venv_python(Path(uv_project_env))
        if candidate.exists():
            return str(candidate)

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidate = _venv_python(Path(virtual_env))
        if candidate.exists():
            return str(candidate)

    start = cwd or Path.cwd()
    for directory in [start, *start.parents]:
        if (directory / "pyproject.toml").is_file():
            candidate = _venv_python(directory / ".venv")
            if candidate.exists():
                return str(candidate)

    return sys.executable


@dataclass(frozen=True)
class PrimeCLIPlugin:
    """Declarative command surface consumed by prime-cli."""

    api_version: int = PRIME_PLUGIN_API_VERSION
    eval_module: str = "verifiers.cli.commands.eval"
    gepa_module: str = "verifiers.cli.commands.gepa"
    install_module: str = "verifiers.cli.commands.install"
    init_module: str = "verifiers.cli.commands.init"
    setup_module: str = "verifiers.cli.commands.setup"
    build_module: str = "verifiers.cli.commands.build"

    def build_module_command(
        self, module_name: str, args: Sequence[str] | None = None
    ) -> list[str]:
        command = [_resolve_workspace_python(), "-m", module_name]
        if args:
            command.extend(args)
        return command


def get_plugin() -> PrimeCLIPlugin:
    """Return the prime plugin definition."""
    return PrimeCLIPlugin()
