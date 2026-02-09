"""Prime-hosted command plugin contract."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from functools import lru_cache
from typing import Sequence

PRIME_PLUGIN_API_VERSION = 1


def _venv_python(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


@lru_cache(maxsize=32)
def _python_can_import_module(
    python_executable: str, module_name: str, cwd: str
) -> bool:
    probe = (
        "import importlib.util, sys; "
        "raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
    )
    try:
        result = subprocess.run(
            [python_executable, "-c", probe, module_name],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _resolve_workspace_python(cwd: Path | None = None) -> str:
    workspace = (cwd or Path.cwd()).resolve()
    workspace_str = str(workspace)
    module = "verifiers.cli.commands.eval"

    def _usable(candidate: Path) -> bool:
        return candidate.exists() and _python_can_import_module(
            str(candidate), module, workspace_str
        )

    uv_project_env = os.environ.get("UV_PROJECT_ENVIRONMENT")
    if uv_project_env:
        candidate = _venv_python(Path(uv_project_env))
        if _usable(candidate):
            return str(candidate)

    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidate = _venv_python(Path(virtual_env))
        if _usable(candidate):
            return str(candidate)

    for directory in [workspace, *workspace.parents]:
        if (directory / "pyproject.toml").is_file():
            candidate = _venv_python(directory / ".venv")
            if _usable(candidate):
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
