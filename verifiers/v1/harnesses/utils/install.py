"""Host-side helpers for installing a harness's program or static binaries into a runtime."""

import asyncio
import hashlib
import io
import shlex
import tarfile
import uuid
from collections.abc import Callable
from weakref import WeakValueDictionary

import httpx

from verifiers.v1.runtimes import Runtime

_ARCHIVE_CACHE: dict[tuple, dict[str, bytes]] = {}
_ARCHIVE_LOCKS: WeakValueDictionary[tuple, asyncio.Lock] = WeakValueDictionary()


async def cached_npm_tarball_members(
    url: str,
    select: Callable[[str], str | None],
    *,
    sha512_b64: str | None = None,
    transform: Callable[[bytes], bytes] | None = None,
) -> dict[str, bytes]:
    """Download an npm tarball once per host event loop and cache selected files."""
    import base64

    key = (url, sha512_b64)
    if key in _ARCHIVE_CACHE:
        return _ARCHIVE_CACHE[key]
    lock = _ARCHIVE_LOCKS.setdefault((asyncio.get_running_loop(), *key), asyncio.Lock())
    async with lock:
        if key in _ARCHIVE_CACHE:
            return _ARCHIVE_CACHE[key]
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
        if sha512_b64 and hashlib.sha512(response.content).digest() != base64.b64decode(
            sha512_b64
        ):
            raise ValueError(f"package checksum mismatch for {url}")

        def extract() -> dict[str, bytes]:
            files: dict[str, bytes] = {}
            with tarfile.open(
                fileobj=io.BytesIO(response.content), mode="r:gz"
            ) as archive:
                for member in archive:
                    rel = select(member.name)
                    if rel is None:
                        continue
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"missing content for {member.name}")
                    raw = source.read()
                    files[rel] = transform(raw) if transform is not None else raw
            return files

        _ARCHIVE_CACHE[key] = await asyncio.to_thread(extract)
        return _ARCHIVE_CACHE[key]


async def stage_executable(runtime: Runtime, target: str, data: bytes) -> None:
    """Atomically write an executable file into a worker sandbox."""
    staged = f"{target}-{uuid.uuid4().hex}"
    await runtime.write(staged, data)
    res = await runtime.run(
        [
            "sh",
            "-c",
            'chmod 755 "$1" && mv -f -- "$1" "$2"',
            "stage-bin",
            staged,
            target,
        ],
        {},
    )
    if res.exit_code:
        raise RuntimeError(f"could not stage {target}: {res.stderr.strip()}")


async def ensure_installed(
    runtime: Runtime,
    *,
    directory: str,
    install: str,
    env: dict[str, str],
    label: str,
    ready: str | None = None,
    lock: str | None = None,
    shell: tuple[str, ...] = ("sh", "-c"),
) -> None:
    """Run `install` in `runtime` under a lock, so concurrent rollouts sharing the runtime
    install once and the rest wait. `ready` is a shell test that skips the install when it
    already holds."""
    cache = runtime.backend.cache
    install_key = (
        f"install:{directory}:"
        + hashlib.sha256(f"{ready or ''}\0{install}".encode()).hexdigest()[:16]
    )
    if ready and install_key in cache.installed:
        return
    lock_q = shlex.quote(lock or f"{directory}/install.lock")
    script = f"({ready}) || ({install})" if ready else install
    run = f"{shlex.join(shell)} {shlex.quote(script)}"
    guarded = (
        f'mkdir -p "$(dirname {lock_q})" && '
        f'if l=$(command -v flock || command -v lockf); then "$l" {lock_q} {run}; else {run}; fi'
    )
    async with cache.setup_lock:
        if ready and install_key in cache.installed:
            return
        if runtime.user is None:
            result = await runtime.run(["sh", "-c", guarded], env)
        else:
            # Install as root, keeping uv's Python open to the agent user.
            python_dir = {
                "UV_PYTHON_INSTALL_DIR": f"{directory}/python",
                "UV_CACHE_DIR": f"{directory}/uv-cache",
            }
            shareable = f"{guarded} && chmod -R a+rX {shlex.quote(directory)}"
            result = await runtime.with_user(None).run(
                ["sh", "-c", shareable], {**python_dir, **env}
            )
        if result.exit_code == 0 and ready:
            cache.installed.add(install_key)
    if result.exit_code != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[-500:]
        raise RuntimeError(f"{label} install failed: {detail}")
