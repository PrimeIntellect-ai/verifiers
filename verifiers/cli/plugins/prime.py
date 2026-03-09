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
WORKSPACE_ENV_DIR = "environments"


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
    workspace_root = _find_workspace_root(workspace)
    workspace_for_probe = workspace_root or workspace
    workspace_str = str(workspace_for_probe)
    module = "verifiers.cli.commands.eval"

    def _usable(candidate: Path) -> bool:
        return candidate.exists() and _python_can_import_module(
            str(candidate), module, workspace_str
        )

    if workspace_root is not None:
        candidate = _venv_python(workspace_root / ".venv")
        if _usable(candidate):
            return str(candidate)

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


def _find_workspace_root(start: Path) -> Path | None:
    for directory in [start, *start.parents]:
        if (
            (directory / "pyproject.toml").is_file()
            and (directory / "verifiers").is_dir()
            and (directory / WORKSPACE_ENV_DIR).is_dir()
        ):
            return directory
    return None


def _current_cwd() -> Path:
    return Path.cwd().resolve()


def _resolve_dir_arg(value: str, cwd: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)

    for directory in [cwd, *cwd.parents]:
        candidate = (directory / path).resolve()
        if candidate.exists() and candidate.is_dir():
            return str(candidate)

    return str((cwd / path).resolve())


def _normalize_or_append_dir_option(
    args: Sequence[str] | None,
    *,
    long_flag: str,
    short_flag: str,
    fallback_value: str | None,
    cwd: Path,
) -> list[str]:
    normalized = list(args or [])
    found_flag = False
    i = 0
    while i < len(normalized):
        token = normalized[i]
        if token in (long_flag, short_flag):
            found_flag = True
            if i + 1 < len(normalized):
                normalized[i + 1] = _resolve_dir_arg(normalized[i + 1], cwd)
                i += 1
        elif token.startswith(f"{long_flag}="):
            found_flag = True
            _, raw_value = token.split("=", 1)
            normalized[i] = f"{long_flag}={_resolve_dir_arg(raw_value, cwd)}"
        elif token.startswith(short_flag) and token != short_flag:
            found_flag = True
            raw_value = token[len(short_flag) :]
            normalized[i] = f"{short_flag}{_resolve_dir_arg(raw_value, cwd)}"
        i += 1

    if not found_flag and fallback_value is not None:
        normalized.extend([long_flag, fallback_value])
    return normalized


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
        cwd = _current_cwd()
        workspace_root = _find_workspace_root(cwd)
        workspace_env_dir: str | None = None
        if workspace_root is not None:
            env_dir = workspace_root / WORKSPACE_ENV_DIR
            if env_dir.is_dir():
                workspace_env_dir = str(env_dir.resolve())

        normalized_args = list(args or [])
        if module_name in (self.install_module, self.build_module, self.init_module):
            normalized_args = _normalize_or_append_dir_option(
                normalized_args,
                long_flag="--path",
                short_flag="-p",
                fallback_value=workspace_env_dir,
                cwd=cwd,
            )
        elif module_name in (self.eval_module, self.gepa_module):
            normalized_args = _normalize_or_append_dir_option(
                normalized_args,
                long_flag="--env-dir-path",
                short_flag="-p",
                fallback_value=workspace_env_dir,
                cwd=cwd,
            )

        command = [_resolve_workspace_python(cwd), "-m", module_name]
        if normalized_args:
            command.extend(normalized_args)
        return command


def get_plugin() -> PrimeCLIPlugin:
    """Return the prime plugin definition."""
    return PrimeCLIPlugin()
