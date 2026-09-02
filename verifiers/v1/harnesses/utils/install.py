"""Host-side helpers for installing a harness's program into a runtime."""

import shlex

from verifiers.v1.runtimes import Runtime

# Linux has flock, macOS/BSD has lockf; either releases the lock when its holder dies.
_LOCKER = '"$(command -v flock || command -v lockf)"'


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
    already holds. The lock is `directory/install.lock` unless the install replaces
    `directory` itself, in which case pass a `lock` path beside it. `shell` runs the script
    (e.g. `bash -o pipefail -c` for pipelines)."""
    lock = lock or f"{directory}/install.lock"
    script = f"{ready} || ({install})" if ready else install
    guarded = (
        f"mkdir -p {shlex.quote(directory)} && {_LOCKER} {shlex.quote(lock)} "
        f"{shlex.join(shell)} {shlex.quote(script)}"
    )
    result = await runtime.run(["sh", "-c", guarded], env)
    if result.exit_code != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeError(f"{label} install failed: {detail}")


async def remove_dir(runtime: Runtime, path: str, label: str) -> None:
    """Delete `path` in `runtime`; a failure names `label` (what the path held)."""
    result = await runtime.run(["rm", "-rf", path], {})
    if result.exit_code != 0:
        raise RuntimeError(
            f"failed to clean up {label}: {result.stderr.strip()[-500:]}"
        )
