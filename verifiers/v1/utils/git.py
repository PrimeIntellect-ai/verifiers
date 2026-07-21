"""Persist an agent's final git patch into `trace.info` at finalize time.

SWE-style tasksets call `capture_patch` from `Task.finalize` — after the harness
finishes, while the runtime is live, before scoring mutates the repo (restoring
test files, switching commits) — so the diff is exactly what the agent produced,
including edits to test files (intentional: they reveal reward hacking).

The diff is taken against `base_commit` when the caller has one — a dataset row
field, or a SHA recorded with `resolve_head` at setup time and kept in host
memory (never the sandbox, where an agent could tamper with it) — so commits the
agent made are included. Bare `HEAD` is the fallback of last resort and misses
agent commits.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.trace import Trace

PATCH_CAP_BYTES = 2_000_000
"""Truncate captured patches beyond this size; `info["patch_truncated"]` marks it."""

# Temp paths are suffixed per invocation with a host-generated nonce: fixed names
# would let concurrent rollouts on a shared-filesystem runtime (subprocess) read
# each other's patches between the write and the read, and would let an agent
# pre-create the predictable path as a FIFO so the shell's redirect blocks the
# rollout forever.
_FULL = "/tmp/vf_agent_patch_full"
_CAPPED = "/tmp/vf_agent_patch"

# `git reset -q` must run even when staging or diffing fails, or the error path
# leaves the tree staged and can break scoring's later checkouts. Every step
# reports: a failure in add, diff, reset, or head fails the capture. A failed
# add (e.g. a stale index.lock from a killed agent git command) leaves a stale
# index, so letting the diff's clean exit stand would record a silently
# incomplete patch; a failed reset (e.g. ENOSPC rewriting .git/index) leaves the
# tree staged, so reporting success would hide a state later scoring may trip
# on; a failed head leaves an empty {capped} (the redirect truncates it before
# head runs), which would read back as a silently empty patch.
_DIFF = (
    "rm -f {full} {capped}; "
    "git add -A; "
    "add_rc=$?; "
    'git -c core.quotepath=off diff --cached --binary "$VF_DIFF_BASE" > {full}; '
    "diff_rc=$?; "
    "git reset -q; "
    "reset_rc=$?; "
    "head -c {cap} {full} > {capped}; "
    "head_rc=$?; "
    "rm -f {full}; "
    "rc=$head_rc; "
    '[ "$diff_rc" -ne 0 ] && rc=$diff_rc; '
    '[ "$reset_rc" -ne 0 ] && rc=$reset_rc; '
    '[ "$add_rc" -ne 0 ] && rc=$add_rc; '
    'exit "$rc"'
)


async def resolve_head(runtime: Runtime, env: dict | None = None) -> str:
    """The repo's current commit SHA, or "" when unresolvable.

    Call at the end of `setup`, before the agent runs, and keep the result in
    host memory (e.g. a dict on the task keyed by `id(runtime)`) for `finalize`
    to pass as `base_commit` — diffing against a pre-agent SHA is what keeps
    commits the agent makes inside the captured patch.
    """
    result = await runtime.run(["git", "rev-parse", "HEAD"], env or {})
    if result.exit_code != 0:
        return ""
    return (result.stdout or "").strip()


async def capture_patch(
    trace: Trace, runtime: Runtime, base_commit: str = "", env: dict | None = None
) -> None:
    """Snapshot the agent's cumulative diff into `trace.info["patch"]`.

    Best-effort by design: a rollout whose sandbox died or whose repo state is
    broken records `info["patch_error"]` instead of failing the rollout —
    scoring still runs and the error stays visible in results.
    """
    nonce = uuid.uuid4().hex
    full, capped = f"{_FULL}_{nonce}", f"{_CAPPED}_{nonce}"
    cmd = _DIFF.format(full=full, capped=capped, cap=PATCH_CAP_BYTES + 1)
    try:
        result = await runtime.run(
            ["sh", "-c", cmd],
            {**(env or {}), "VF_DIFF_BASE": base_commit or "HEAD"},
        )
        if result.exit_code != 0:
            trace.info["patch_error"] = (
                f"exit={result.exit_code} {(result.stderr or '').strip()[-500:]}"
            )
            return
        raw = await runtime.read(capped)
    except Exception as exc:  # noqa: BLE001 - capture must never fail the rollout.
        trace.info["patch_error"] = f"{type(exc).__name__}: {exc}"
        return
    finally:
        # Unique names don't overwrite each other, so leftovers would accumulate
        # on shared-filesystem runtimes; removal is best-effort by design.
        try:
            await runtime.run(["rm", "-f", full, capped], env or {})
        except Exception:  # noqa: BLE001,S110 - cleanup must never fail the rollout.
            pass
    if len(raw) > PATCH_CAP_BYTES:
        raw = raw[:PATCH_CAP_BYTES]
        trace.info["patch_truncated"] = True
    trace.info["patch"] = raw.decode("utf-8", errors="replace")
