"""Transfer package inputs; execute build backends only in the server's runtime."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import re
import shlex
import subprocess
import sysconfig
import tarfile
import tomllib
import uuid
from collections.abc import AsyncIterator
from functools import cache
from importlib.metadata import PackageNotFoundError, PathDistribution, version
from pathlib import Path

from verifiers.v1.errors import ToolsetError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import clip, get_logger

logger = get_logger(__name__)

_BUILD_TIMEOUT = 300
_SOURCE_LIMIT = 256 * 1024 * 1024
_EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}

# RuntimeProcess signals may target only the immediate process. This supervisor
# owns a separate process group for the build and reaps it before acknowledging
# termination, including when a backend leaves children behind after exiting.
_BUILD_SUPERVISOR = """
import contextlib, os, signal, subprocess, sys

stopping = False
def stop(signum, frame):
    global stopping
    stopping = True

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
child = subprocess.Popen(
    ["sh", "-c", sys.argv[1]], stdin=subprocess.DEVNULL, start_new_session=True
)
try:
    while child.poll() is None:
        if stopping:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        try:
            child.wait(timeout=0.1)
        except subprocess.TimeoutExpired:
            pass
finally:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait()
sys.exit(143 if stopping else child.returncode)
"""


def _git(directory: Path, *args: str) -> subprocess.CompletedProcess:
    # File discovery only: no build hooks, Git hooks, or fsmonitor programs on the host.
    return subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(directory), *args],
        capture_output=True,
        timeout=30,
        check=False,
    )


def _source_files(src: Path) -> tuple[Path, list[Path]]:
    """Keep repository/workspace-relative inputs and uv configuration together."""
    root = src
    try:
        result = _git(src, "rev-parse", "--show-toplevel")
        if result.returncode == 0:
            root = Path(os.fsdecode(result.stdout).strip()).resolve()
            result = _git(
                root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"
            )
            if result.returncode:
                raise ToolsetError(f"cannot list package sources in {root}")
            files = list(
                dict.fromkeys(
                    root / os.fsdecode(name)
                    for name in result.stdout.split(b"\0")
                    if name
                )
            )
            # uv uses these even when a checkout deliberately ignores them.
            for parent in (src, *src.parents):
                files.extend(
                    parent / name
                    for name in ("pyproject.toml", "uv.toml", ".python-version")
                    if (parent / name).is_file()
                )
                if parent == root:
                    break
            return root, list(dict.fromkeys(files))
    except FileNotFoundError:
        pass
    # Source distributions need no Git. A uv workspace can also be an unpacked tree.
    for parent in src.parents:
        project = parent / "pyproject.toml"
        if project.is_file():
            config = tomllib.loads(project.read_text())
            if "workspace" in config.get("tool", {}).get("uv", {}):
                root = parent
    paths = []
    for directory, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in _EXCLUDED)
        paths.extend(Path(directory) / name for name in sorted(files))
        paths.extend(
            Path(directory) / name
            for name in dirs
            if (Path(directory) / name).is_symlink()
        )
    return root, paths


@cache
def package_source(src: Path) -> tuple[bytes, str, dict[str, str]]:
    """Snapshot source files or an installed wheel without executing package code."""
    buffer = io.BytesIO()
    env: dict[str, str] = {}
    size = 0
    with tarfile.open(fileobj=buffer, mode="w:gz", dereference=False) as archive:
        if src.suffix == ".dist-info":
            # Installed RECORD entries may be symlinks into uv's wheel cache.
            # Transfer their contents, not links to the evaluator's filesystem.
            archive.dereference = True
            dist = PathDistribution(src)
            files = dist.files
            if not files:
                raise ToolsetError(f"cannot package {src}: missing RECORD")
            scripts = {
                entry.name
                for entry in dist.entry_points
                if entry.group in {"console_scripts", "gui_scripts"}
            }
            selected = []
            for file in files:
                if ".." in file.parts or file.is_absolute():
                    if (
                        file.name in scripts
                        and Path(file.locate()).parent.resolve()
                        == Path(sysconfig.get_path("scripts")).resolve()
                    ):
                        continue
                    raise ToolsetError(
                        f"cannot package {src}: installed file outside site-packages: {file}"
                    )
                if file.suffix == ".pyc" or (
                    file.parent.name == src.name
                    and file.name
                    in {
                        "RECORD",
                        "RECORD.jws",
                        "RECORD.p7s",
                        "INSTALLER",
                        "REQUESTED",
                        "direct_url.json",
                        "uv_cache.json",
                    }
                ):
                    continue
                selected.append((Path(file.locate()), Path("package") / file))
            root, directory = src.parent, "package"
        else:
            root, files = _source_files(src)
            directory = "source/" + src.relative_to(root).as_posix()
            selected = []
            for path in files:
                relative = path.relative_to(root)
                if any(part in _EXCLUDED for part in relative.parts) or path.suffix in {
                    ".pyc",
                    ".pyo",
                }:
                    continue
                if not path.exists() and not path.is_symlink():
                    continue  # A tracked file deleted in this checkout.
                selected.append((path, Path("source") / relative))
                if path.name == "pyproject.toml" and path.is_file():
                    project = tomllib.loads(path.read_text()).get("project", {})
                    name = project.get("name")
                    if name and "version" in project.get("dynamic", []):
                        # The snapshot has no Git database. Preserve the installed
                        # distribution's version for hatch-vcs/setuptools-scm builds.
                        with contextlib.suppress(PackageNotFoundError):
                            key = re.sub(r"[-_.]+", "_", name).upper()
                            env[f"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_{key}"] = version(
                                name
                            )
        for path, relative in selected:
            if src.suffix != ".dist-info" and not path.resolve().is_relative_to(root):
                raise ToolsetError(f"package input escapes {root}: {path}")
            info = archive.gettarinfo(str(path), arcname=relative.as_posix())
            if not (info.isfile() or info.isdir() or info.issym() or info.islnk()):
                raise ToolsetError(f"unsupported package input: {path}")
            if info.issym():
                # Absolute host links become relative links in the staged tree.
                info.linkname = os.path.relpath(path.resolve(), path.parent)
            size += info.size
            if size > _SOURCE_LIMIT:
                raise ToolsetError(f"package inputs for {src} exceed 256 MiB")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if info.isfile():
                with path.open("rb") as file:
                    archive.addfile(info, file)
            else:
                archive.addfile(info)
    return buffer.getvalue(), directory, env


async def _build(
    runtime: Runtime, command: str, env: dict[str, str], *, python: str = "python3"
) -> str:
    """Bound build time and kill the runtime process before releasing its files."""
    process = None

    async def tail(stream) -> bytes:
        output = b""
        async for chunk in stream:
            output = (output + chunk)[-2000:]
        return output

    try:
        async with asyncio.timeout(_BUILD_TIMEOUT):
            process = await runtime.exec(
                [python, "-c", _BUILD_SUPERVISOR, command], env
            )
            stdout, stderr, code = await asyncio.gather(
                tail(process.stdout), tail(process.stderr), process.wait()
            )
    except BaseException:
        if process is not None:

            async def stop() -> None:
                try:
                    async with asyncio.timeout(10):
                        await process.terminate()
                        await process.wait()
                except Exception as exc:
                    logger.warning(
                        "Failed to terminate package build, sending SIGKILL - %s: %s",
                        type(exc).__name__,
                        clip(str(exc)),
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )
                    await process.kill()
                    await process.wait()

            try:
                await run_shielded(stop())
            except Exception as exc:
                logger.warning(
                    "Failed to stop package build - %s: %s",
                    type(exc).__name__,
                    clip(str(exc)),
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
        raise
    if code:
        detail = (stdout + b"\n" + stderr).decode(errors="replace").strip()
        raise ToolsetError(f"package build failed in runtime: {detail[-2000:]}")
    lines = stdout.decode(errors="replace").strip().splitlines()
    return lines[-1] if lines else ""


@contextlib.asynccontextmanager
async def built_package(
    runtime: Runtime, src: Path, root: str, env: dict[str, str], *, python: str
) -> AsyncIterator[str]:
    """Lend a wheel built in this runtime, cleaning up its inputs on every exit."""
    data, directory, version_env = await asyncio.to_thread(
        package_source, src.resolve()
    )
    build = f"{root}/build-{uuid.uuid4().hex}"
    archive = f"{build}/source.tar.gz"
    output = f"{build}/dist"
    quote = shlex.quote
    command = (
        f"set -e; mkdir -p {quote(output)}; "
        f"tar -xzf {quote(archive)} -C {quote(build)}; "
        f"cd {quote(f'{build}/{directory}')}; "
        'export PATH="$HOME/.local/bin:$PATH"; '
    )
    if src.suffix == ".dist-info":
        command += f"uv tool run --python {quote(python)} --from wheel==0.48.0 wheel pack --dest-dir {quote(output)} .; "
    else:
        command += f"uv build --wheel --python {quote(python)} --no-create-gitignore --color never --out-dir {quote(output)} .; "
    command += (
        f"set -- {quote(output)}/*.whl; "
        '[ "$#" -eq 1 ] && [ -f "$1" ] || { echo "expected one built wheel" >&2; exit 1; }; '
        'printf "%s\\n" "$1"'
    )
    try:
        await run_shielded(runtime.write(archive, data))
        try:
            wheel = await _build(
                runtime, command, {**env, **version_env}, python=python
            )
        except TimeoutError as exc:
            raise ToolsetError(
                f"package build timed out after {_BUILD_TIMEOUT}s for {src}"
            ) from exc
        if not wheel.startswith(output + "/") or Path(wheel).name != wheel.removeprefix(
            output + "/"
        ):
            raise ToolsetError(
                f"package build for {src} returned an invalid wheel path"
            )
        yield wheel
    finally:
        try:
            result = await run_shielded(runtime.run(["rm", "-rf", build], {}))
            if result.exit_code:
                raise ToolsetError(result.stderr or result.stdout)
        except Exception as exc:
            logger.warning(
                "Failed to clean up package build files - %s: %s",
                type(exc).__name__,
                clip(str(exc)),
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
