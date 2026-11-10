"""Host-side helpers for installing a harness's program into a runtime."""

import shlex
import time

from verifiers.v1.runtimes import Runtime
from verifiers.v1.utils.format import format_time
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)


class HarnessNotInstalled(RuntimeError):
    """A harness configured with `install="never"` found its program missing or broken."""


INSTALLED_MARKER = "vf-install-ran-8f3c1d"
"""Printed after a successful install, so the host can tell it from a `ready` skip."""


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
    never: bool = False,
    check: str | None = None,
) -> bool:
    """Run `install` in `runtime` under a lock, so concurrent rollouts sharing the runtime
    install once and the rest wait. `ready` is a shell test that skips the install when it
    already holds. The lock is `directory/install.lock` unless the install replaces
    `directory` itself, in which case pass a `lock` path beside it. `shell` runs the script
    (e.g. `bash -o pipefail -c` for pipelines).

    Returns whether `install` ran (False when `ready` held). With a `ready` check, an
    install is logged at INFO with its duration, and at WARNING when this runtime had
    already completed it: something deleted or broke a working install, so it was
    reinstalled. Without one, `install` runs every time and checks for itself, so it
    returns True and logs nothing.

    `never` (the harness config's `install="never"`) installs nothing: it runs `check`
    (by default `ready`; an install without `ready` must give one) and raises
    `HarnessNotInstalled` naming `directory` and the check when it fails.

    The lock is `flock` (Linux, busybox) or `lockf` (macOS/BSD), both released when the holder
    dies; an image with neither falls back to a symlink spinlock whose owner is recorded as
    `pid:starttime` so a dead holder is reaped even if its pid was reused."""
    if never:
        test = check or ready
        if test is None:
            raise ValueError(f"{label} has no install check for install='never'")
        result = await runtime.with_user(None).run(["sh", "-c", test], env)
        if result.exit_code != 0:
            raise HarnessNotInstalled(
                f"{label} is not installed in {directory} and install='never' "
                f"(check failed: {test})"
            )
        return False
    lock = shlex.quote(lock or f"{directory}/install.lock")
    ran = f"echo {INSTALLED_MARKER}"
    script = (
        f"{ready} || {{ ({install}) && {ran}; }}" if ready else f"({install}) && {ran}"
    )
    run = f"{shlex.join(shell)} {shlex.quote(script)}"
    # The owner token is pid:starttime (starttime from /proc where readable), so a reused pid
    # does not pass as the live holder. A lock that is not a symlink (left behind by
    # flock/lockf) or whose owner is gone is stale and reaped; `kill -0 ""` succeeds under
    # busybox ash, hence the explicit -z.
    ident = (
        'ident() { if [ -r "/proc/$1/stat" ] && s=$(sed "s/^.*) //" "/proc/$1/stat" '
        '| cut -d" " -f20) && [ -n "$s" ]; then echo "$1:$s"; else echo "$1"; fi; }; '
        "me=$(ident $$); "
    )
    spinlock = (
        f"{ident}"
        f'until ln -s "$me" {lock} 2>/dev/null; do owner=$(readlink {lock} 2>/dev/null); '
        f'pid=${{owner%%:*}}; if [ -z "$owner" ] || ! kill -0 "$pid" 2>/dev/null '
        f'|| [ "$(ident "$pid")" != "$owner" ]; then '
        f'[ "$(readlink {lock} 2>/dev/null)" != "$owner" ] || rm -f {lock}; fi; '
        f"sleep 0.1; done; "
        f'trap \'[ "$(readlink {lock} 2>/dev/null)" != "$me" ] || rm -f {lock}\' EXIT; {run}'
    )
    guarded = (
        f'mkdir -p "$(dirname {lock})" && '
        f'if l=$(command -v flock || command -v lockf); then "$l" {lock} {run}; else {spinlock}; fi'
    )
    start = time.monotonic()
    if runtime.user is None:
        result = await runtime.run(["sh", "-c", guarded], env)
    else:
        # Install as root (system packages), keeping uv's Python in `directory`, then
        # open it to the agent user: some installers leave it private.
        python_dir = {
            "UV_PYTHON_INSTALL_DIR": f"{directory}/python",
            "UV_CACHE_DIR": f"{directory}/uv-cache",
        }
        shareable = f"{guarded} && chmod -R a+rX {shlex.quote(directory)}"
        result = await runtime.with_user(None).run(
            ["sh", "-c", shareable], {**python_dir, **env}
        )
    if result.exit_code != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[-500:]
        raise RuntimeError(f"{label} install failed: {detail}")
    installed = INSTALLED_MARKER in result.stdout
    if ready is None:
        return installed
    completed = runtime.backend.cache.installed
    if installed and directory in completed:
        logger.warning(
            "Reinstalled %s after its install in this runtime stopped passing its check "
            "in %s directory=%s",
            label,
            format_time(time.monotonic() - start),
            directory,
        )
    elif installed:
        logger.info(
            "Installed %s in %s directory=%s",
            label,
            format_time(time.monotonic() - start),
            directory,
        )
    completed.add(directory)
    return installed


async def remove_dir(runtime: Runtime, path: str, label: str) -> None:
    """Delete `path` in `runtime`; a failure names `label` (what the path held)."""
    result = await runtime.run(["rm", "-rf", path], {})
    if result.exit_code != 0:
        raise RuntimeError(
            f"failed to clean up {label}: {result.stderr.strip()[-500:]}"
        )
