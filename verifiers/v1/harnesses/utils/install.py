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
    bootstrap_curl: bool = True,
) -> None:
    """Run `install` in `runtime` under a lock, so concurrent rollouts sharing the runtime
    install once and the rest wait. `ready` is a shell test that skips the install when it
    already holds. The lock is `directory/install.lock` unless the install replaces
    `directory` itself, in which case pass a `lock` path beside it. `shell` runs the script
    (e.g. `bash -o pipefail -c` for pipelines).
    Curl's own installer disables `bootstrap_curl` to avoid bootstrapping itself.

    The lock is `flock` (Linux, busybox) or `lockf` (macOS/BSD), both released when the holder
    dies; an image with neither falls back to a symlink spinlock whose owner is recorded as
    `pid:starttime` so a dead holder is reaped even if its pid was reused."""
    lock = shlex.quote(lock or f"{directory}/install.lock")
    script = f"{ready} || ({install})" if ready else install
    # The owner token is pid:starttime (starttime from /proc where readable), so a reused pid
    # does not pass as the live holder. A lock that is not a symlink (left behind by
    # flock/lockf) or whose owner is gone is stale and reaped; `kill -0 ""` succeeds under
    # busybox ash, hence the explicit -z.
    ident = (
        'ident() { if [ -r "/proc/$1/stat" ] && s=$(sed "s/^.*) //" "/proc/$1/stat" '
        '| cut -d" " -f20) && [ -n "$s" ]; then echo "$1:$s"; else echo "$1"; fi; }; '
        "me=$(ident $$); "
    )
    cleanup = f'[ "$(readlink {lock} 2>/dev/null)" != "$me" ] || rm -f {lock}'
    spinlock = (
        f"{ident}"
        f'until ln -s "$me" {lock} 2>/dev/null; do owner=$(readlink {lock} 2>/dev/null); '
        f'pid=${{owner%%:*}}; if [ -z "$owner" ] || ! kill -0 "$pid" 2>/dev/null '
        f'|| [ "$(ident "$pid")" != "$owner" ]; then '
        f'[ "$(readlink {lock} 2>/dev/null)" != "$owner" ] || rm -f {lock}; fi; '
        f"sleep 0.1; done; "
        f'trap {shlex.quote(cleanup)} EXIT; "$@"'
    )
    guarded = (
        f"mkdir -p {shlex.quote(directory)} && "
        f'if l=$(command -v flock || command -v lockf); then "$l" {lock} "$@"; else {spinlock}; fi'
    )
    argv = ["sh", "-c", guarded, "vf-install", *shell]
    if runtime.user is not None and bootstrap_curl:
        if ready and (await runtime.run([*argv, ready], env)).exit_code == 0:
            return
        # Curl has its own root-owned filesystem lock across all user caches.
        await runtime.ensure_curl()
    result = await runtime.run([*argv, script], env)
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
