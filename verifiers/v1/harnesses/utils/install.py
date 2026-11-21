"""Host-side helpers for installing a harness's program into a runtime."""

import hashlib
import shlex

from verifiers.v1.runtimes import Runtime


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


async def remove_dir(runtime: Runtime, path: str, label: str) -> None:
    """Delete `path` in `runtime`; a failure names `label` (what the path held)."""
    result = await runtime.run(["rm", "-rf", path], {})
    if result.exit_code != 0:
        raise RuntimeError(
            f"failed to clean up {label}: {result.stderr.strip()[-500:]}"
        )
