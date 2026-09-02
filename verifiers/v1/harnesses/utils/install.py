"""Host-side helpers for installing a harness's program into a runtime."""

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
    already holds. The lock is `directory/install.lock` unless the install replaces
    `directory` itself, in which case pass a `lock` path beside it. `shell` runs the script
    (e.g. `bash -o pipefail -c` for pipelines).

    The lock is `flock` (Linux, busybox) or `lockf` (macOS/BSD), both released when the holder
    dies; an image with neither falls back to a symlink spinlock that reaps a dead owner."""
    lock = shlex.quote(lock or f"{directory}/install.lock")
    script = f"{ready} || ({install})" if ready else install
    run = f"{shlex.join(shell)} {shlex.quote(script)}"
    # A lock that is not a symlink (left behind by flock/lockf) or whose owner pid is gone
    # is stale and reaped. `kill -0 ""` succeeds under busybox ash, hence the explicit -z.
    spinlock = (
        f'until ln -s "$$" {lock} 2>/dev/null; do owner=$(readlink {lock} 2>/dev/null); '
        f'if [ -z "$owner" ] || ! kill -0 "$owner" 2>/dev/null; then '
        f'[ "$(readlink {lock} 2>/dev/null)" != "$owner" ] || rm -f {lock}; fi; '
        f"sleep 0.1; done; "
        f'trap \'[ "$(readlink {lock} 2>/dev/null)" != "$$" ] || rm -f {lock}\' EXIT; {run}'
    )
    guarded = (
        f"mkdir -p {shlex.quote(directory)} && "
        f'if l=$(command -v flock || command -v lockf); then "$l" {lock} {run}; else {spinlock}; fi'
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
