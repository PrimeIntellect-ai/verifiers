"""Copy MCP package inputs and install them inside the server's runtime."""

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
from functools import cache
from importlib.metadata import PackageNotFoundError, PathDistribution, version
from pathlib import Path

from verifiers.v1.errors import ToolsetError
from verifiers.v1.harnesses.utils.install import remove_dir
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import _drain
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import clip, get_logger

logger = get_logger(__name__)
_BUILD_TIMEOUT = 300
# uv fetches Git dependencies with the runtime's `git`, which slim images lack.
_ENSURE_GIT = (
    "command -v git >/dev/null 2>&1 "
    "|| { apt-get update -qq && apt-get install -y -qq git; } "
    "|| apk add --no-cache git "
    "|| { echo 'a Git dependency needs git, which this runtime cannot install' >&2; "
    "exit 1; }"
)
_EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    ".cache",
    "build",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}

# Prime signals only the immediate process. Keep build/install children in a
# separate group and reap that group before acknowledging cancellation.
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
    while child.poll() is None and not stopping:
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


def _source_files(src: Path) -> tuple[Path, list[Path]]:
    # A package's repository may contain unrelated datasets or other projects.
    # Only an explicit uv workspace expands the package's source boundary.
    root = src
    for parent in src.parents:
        project = parent / "pyproject.toml"
        if project.is_file() and "workspace" in tomllib.loads(project.read_text()).get(
            "tool", {}
        ).get("uv", {}):
            root = parent
            break

    def collect(directory: Path) -> list[Path]:
        try:
            listing = subprocess.check_output(
                [
                    "git",
                    "-c",
                    "core.fsmonitor=false",
                    "-C",
                    str(directory),
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "--",
                    ".",
                ],
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            files = []
            for current, dirs, names in os.walk(directory):
                dirs[:] = [name for name in dirs if name not in _EXCLUDED]
                files.extend(
                    Path(current) / name
                    for name in dirs
                    if (Path(current) / name).is_symlink()
                )
                files.extend(Path(current) / name for name in names)
            return files
        files = []
        for name in listing.split(b"\0"):
            if not name:
                continue
            path = directory / os.fsdecode(name)
            if path.is_dir() and not path.is_symlink():
                # Git lists a submodule as one directory. Recurse separately so
                # its untracked and modified files are included as well.
                if not (path / ".git").exists():
                    raise ToolsetError(
                        f"initialize package submodule before building: {path}"
                    )
                files.extend(collect(path))
            else:
                files.append(path)
        return files

    return root, collect(root)


def _git_requirement(path: Path) -> bool:
    """Whether a `pyproject.toml` or wheel `METADATA` requires a dependency from Git."""
    if path.name == "METADATA":
        return any(
            line.startswith("Requires-Dist:") and "git+" in line
            for line in path.read_text(errors="replace").splitlines()
        )
    data = tomllib.loads(path.read_text())
    project = data.get("project", {})
    requirements = [
        *project.get("dependencies", []),
        *(
            requirement
            for group in project.get("optional-dependencies", {}).values()
            for requirement in group
        ),
    ]
    sources = [
        source
        for entry in data.get("tool", {}).get("uv", {}).get("sources", {}).values()
        for source in (entry if isinstance(entry, list) else [entry])
    ]
    return any("git+" in requirement for requirement in requirements) or any(
        isinstance(source, dict) and "git" in source for source in sources
    )


@cache
def package_source(src: Path) -> tuple[bytes, str, dict[str, str], bool]:
    """Snapshot source or installed files without executing a host build hook, and
    report whether installing them fetches a dependency from Git."""
    env = {}
    git = False
    if src.suffix == ".dist-info":
        dist = PathDistribution(src)
        if not dist.files:
            raise ToolsetError(f"cannot package {src}: missing RECORD")
        root, directory = src.parent, "."
        files = []
        for entry in dist.files:
            if entry.is_absolute() or ".." in entry.parts:
                # Entry-point scripts are regenerated by the sandbox's installer.
                if (
                    entry.name in {item.name for item in dist.entry_points}
                    and Path(entry.locate()).parent.resolve()
                    == Path(sysconfig.get_path("scripts")).resolve()
                ):
                    continue
                raise ToolsetError(
                    f"installed package file outside site-packages: {entry}"
                )
            if entry.suffix != ".pyc" and not (
                entry.parent.name == src.name
                and entry.name
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
                files.append(Path(entry.locate()))
    else:
        root, files = _source_files(src)
        directory = src.relative_to(root).as_posix()
    buffer, size = io.BytesIO(), 0
    with tarfile.open(
        fileobj=buffer, mode="w:gz", dereference=src.suffix == ".dist-info"
    ) as archive:
        for path in dict.fromkeys(files):
            relative = path.relative_to(root)
            if _EXCLUDED.intersection(relative.parts) or not path.exists():
                continue
            if path.name == "pyproject.toml" or (
                path.name == "METADATA" and path.parent.suffix == ".dist-info"
            ):
                git = git or _git_requirement(path)
            if src.suffix != ".dist-info":
                if not path.resolve().is_relative_to(root):
                    raise ToolsetError(f"package input escapes {root}: {path}")
                if path.name == "pyproject.toml":
                    project = tomllib.loads(path.read_text()).get("project", {})
                    if "version" in project.get("dynamic", []):
                        with contextlib.suppress(PackageNotFoundError):
                            name = project["name"]
                            env[
                                "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_"
                                + re.sub(r"[-_.]+", "_", name).upper()
                            ] = version(name)

            def include(info, path=path):
                nonlocal size
                if not (info.isfile() or info.issym()):
                    raise ToolsetError(f"unsupported package input: {path}")
                if info.issym():
                    info.linkname = os.path.relpath(path.resolve(), path.parent)
                size += info.size
                if size > 256 * 1024 * 1024:
                    raise ToolsetError(f"package inputs for {src} exceed 256 MiB")
                return info

            archive.add(path, arcname=str(relative), recursive=False, filter=include)
    return buffer.getvalue(), directory, env, git


async def _build(
    runtime: Runtime, command: str, env: dict[str, str], *, python: str = "python3"
) -> None:
    process = None
    completion = None
    try:
        async with asyncio.timeout(_BUILD_TIMEOUT):
            process = await runtime.exec(
                [python, "-c", _BUILD_SUPERVISOR, command], env
            )
            completion = asyncio.gather(
                _drain(process.stdout), _drain(process.stderr), process.wait()
            )
            stdout, stderr, code = await asyncio.shield(completion)
        if code:
            detail = (stdout + b"\n" + stderr).decode(errors="replace").strip()
            raise ToolsetError(f"package install failed in runtime: {detail[-2000:]}")
    finally:
        if process is not None:

            async def stop():
                if await process.poll() is None:
                    try:
                        async with asyncio.timeout(10):
                            await process.terminate()
                            await process.wait()
                    except Exception as exc:
                        logger.warning(
                            "Failed to terminate package install, sending SIGKILL - %s: %s",
                            type(exc).__name__,
                            clip(str(exc)),
                            exc_info=logger.isEnabledFor(logging.DEBUG),
                        )
                        await process.kill()
                        await process.wait()
                if completion is not None:
                    await completion

            await run_shielded(stop())


async def install_package(
    runtime: Runtime, src: Path, root: str, env: dict[str, str], *, python: str
) -> None:
    data, directory, version_env, git = await asyncio.to_thread(
        package_source, src.resolve()
    )
    build = f"{root}/build-{uuid.uuid4().hex}"
    quote = shlex.quote
    command = (
        f"set -e; mkdir -p {quote(build + '/source')}; "
        f"tar -xzf {quote(build + '/source.tar.gz')} -C {quote(build + '/source')}; "
        f"cd {quote(build + '/source/' + directory)}; "
        'export PATH="$HOME/.local/bin:$PATH"; '
    )
    if git:
        command += f"{_ENSURE_GIT}; "
    target = "."
    if src.suffix == ".dist-info":
        output = quote(build + "/dist")
        command += (
            f"mkdir -p {output}; "
            f"uv tool run --python {quote(python)} --from wheel==0.48.0 wheel pack --dest-dir {output} .; "
        )
        target = f"{quote(build + '/dist')}/*.whl"
    command += f"uv pip install --no-editable --python {quote(python)} {target}"
    try:
        await run_shielded(runtime.write(f"{build}/source.tar.gz", data))
        await _build(runtime, command, {**env, **version_env}, python=python)
    except TimeoutError as exc:
        raise ToolsetError(
            f"package install timed out after {_BUILD_TIMEOUT}s for {src}"
        ) from exc
    finally:
        try:
            await run_shielded(remove_dir(runtime, build, "MCP package inputs"))
        except Exception as exc:
            logger.warning(
                "Failed to clean up package inputs - %s: %s",
                type(exc).__name__,
                clip(str(exc)),
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
