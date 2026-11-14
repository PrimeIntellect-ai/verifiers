"""Persist an agent's final git patch into `trace.info` at finalize time.

SWE-style tasksets call `capture_patch` from `Task.finalize` — after the harness
finishes, while the runtime is live, before scoring mutates the repo (restoring
test files, switching commits) — so the diff is exactly what the agent produced,
including edits to test files (intentional: they reveal reward hacking), and
excluding whatever `snapshot_untracked` recorded before the agent started.

The diff is taken against `base_commit` when the caller has one — a dataset row
field, or a SHA recorded with `resolve_head` at setup time and kept in host
memory (never the sandbox, where an agent could tamper with it) — so commits the
agent made are included. Bare `HEAD` is the fallback of last resort and misses
agent commits.
"""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from verifiers.v1.errors import SandboxError

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
#
# The unstage step is deliberately outside that accounting: if it fails the patch is
# merely as wide as it used to be, which is a worse patch, not a broken rollout. The
# `$#` test is load-bearing — `git reset -q --` with no pathspec unstages everything.
_DIFF = (
    "rm -f {full} {capped}; "
    "git add -A; "
    "add_rc=$?; "
    '[ "$#" -gt 0 ] && git reset -q -- "$@"; '
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


async def snapshot_untracked(runtime: Runtime, env: dict | None = None) -> list[str]:
    """The repo's untracked files, to hand `capture_patch` as `ignore`.

    Call at the end of `setup`, before the agent runs, and keep the result in host
    memory beside `resolve_head`'s SHA. Whatever it lists came with the image, so the
    agent cannot be credited with it, and a patch that carries it fails `git apply` in
    a fresh container of that same image — which is what an isolated grading box is.

    Sandbox snapshotting will make this free: once runtimes can snapshot and diff a
    filesystem, the pre-agent untracked set falls out of the diff with no setup-side
    bookkeeping in any taskset. Drop this then.
    """
    result = await runtime.run(
        ["sh", "-c", "git ls-files --others --exclude-standard -z"], env or {}
    )
    if result.exit_code != 0:
        return []
    stdout = result.stdout or ""
    # Runtimes decode stdout with errors="replace", so a filename that isn't
    # valid UTF-8 reaches us with U+FFFD substituted — it can never match the
    # real file again, and capture_patch would credit the agent with an image
    # file it didn't create. A wrong patch is worse than a loud setup failure.
    # (A filename legitimately containing U+FFFD trips this too; that is the
    # price of the runtimes' lossy text contract.)
    if "�" in stdout:
        raise ValueError(
            "snapshot_untracked: `git ls-files` returned a filename that is not "
            "valid UTF-8; it cannot round-trip through the runtime's text "
            "decoding, so the pre-agent untracked set would be wrong. Rename "
            "the offending file in the image or exclude it via .gitignore."
        )
    return [path for path in stdout.split("\0") if path]


async def capture_patch(
    trace: Trace,
    runtime: Runtime,
    base_commit: str = "",
    env: dict | None = None,
    write_path: str | None = None,
    ignore: list[str] | None = None,
) -> None:
    """Snapshot the agent's cumulative diff into `trace.info["patch"]`.

    `ignore` names paths to leave out — pass `snapshot_untracked`'s list from setup, or
    `git add -A` credits the agent with untracked files the image shipped. R2E-Gym boxes
    ship three (`datasets`, `install.sh`, `run_tests.sh`), and a patch carrying them
    fails `git apply` in a fresh container of that very image — which is what an
    isolated grading box is.

    Two failure modes, attributed differently, because they deserve different outcomes.

    A non-zero exit means the box answered and git refused: a stale `index.lock` from a
    killed agent command, a deleted `.git`, a `base_commit` the agent rewrote out of
    existence, a disk it filled. That is the agent's own environment, so it records
    `info["patch_error"]` and lets the rollout score — a run with no patch grades as a
    run that changed nothing, which is the right reward.

    An exception means the box never answered: the sandbox died, the exec timed out,
    the transport dropped. Nothing there is the policy's doing, so it raises. Scoring
    the rollout anyway would feed a zero to training that says only that our
    infrastructure failed.

    `write_path` additionally writes the patch to that path inside the box, for tasks
    graded in a second sandbox. Point it at `vf.ARTIFACTS_DIR` (e.g.
    `/logs/artifacts/patch.diff`) and collection picks it up with no declaration. Leave
    it on the convention sweep rather than declaring it as an `Artifact`: a declared
    path is collected strictly, which would turn an agent-broken repo into a rollout
    error instead of the low score it should earn.
    """
    nonce = uuid.uuid4().hex
    full, capped = f"{_FULL}_{nonce}", f"{_CAPPED}_{nonce}"
    cmd = _DIFF.format(full=full, capped=capped, cap=PATCH_CAP_BYTES + 1)
    # `git reset -- <path>` treats arguments as pathspecs, so a file named
    # `*.py` would unstage every Python file the agent touched. `:(literal)`
    # makes git match the name byte-for-byte (and also neutralizes names that
    # start with `:`, which git would otherwise parse as pathspec magic).
    ignore_pathspecs = [f":(literal){path}" for path in ignore or []]
    try:
        result = await runtime.run(
            ["sh", "-c", cmd, "vf-capture-patch", *ignore_pathspecs],
            {**(env or {}), "VF_DIFF_BASE": base_commit or "HEAD"},
        )
        if result.exit_code != 0:
            # Not every runtime raises when the box is gone — Docker returns `docker
            # exec`'s own non-zero result, which is indistinguishable from git failing.
            # One probe on the failure path tells the two apart before we blame anyone.
            if (await runtime.run(["true"], {})).exit_code != 0:
                raise SandboxError(
                    f"patch capture failed and the box stopped answering: "
                    f"{(result.stderr or '').strip()[-300:]}"
                )
            trace.info["patch_error"] = (
                f"exit={result.exit_code} {(result.stderr or '').strip()[-500:]}"
            )
            return
        raw = await runtime.read(capped)
    finally:
        # Unique names don't overwrite each other, so leftovers would accumulate
        # on shared-filesystem runtimes; removal is best-effort by design.
        try:
            await runtime.run(["rm", "-f", full, capped], env or {})
        except Exception:  # noqa: BLE001, S110 - cleanup must never fail the rollout
            pass
    if len(raw) > PATCH_CAP_BYTES:
        raw = raw[:PATCH_CAP_BYTES]
        trace.info["patch_truncated"] = True
    trace.info["patch"] = raw.decode("utf-8", errors="replace")
    if write_path is not None:
        parent = str(PurePosixPath(write_path).parent)
        await runtime.run(["mkdir", "-p", parent], env or {})
        await runtime.write(write_path, raw)
