"""Run code an agent delivered while grading, without the grader's privileges, and make
sure nothing it started outlives the block."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import ProgramResult, RuntimeProcess
from verifiers.v1.runtimes.runtime import Runtime

if TYPE_CHECKING:
    from verifiers.v1.task import Task

STARTUP_HOOKS = ("PYTHONSTARTUP", "BASH_ENV", "ENV")
"""Variables that make a shell or Python run code before the command it was given."""

DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Each command runs in a new session. `setsid` started from a background subshell, which
# is no process group leader, calls setsid(2) without forking (busybox has no `-w`), and
# `wait` returns its exit code; stdin is passed explicitly, since a background list
# would get /dev/null. A root view records the session, since only those can be told
# apart from the grader's own processes; the record file is the first argument.
_SESSION = (
    'f=$1; shift; command -v setsid >/dev/null || { echo "untrusted: needs setsid" >&2; exit 127; }; '
    'exec 3<&0; (exec setsid sh -c \'[ -z "$0" ] || echo $$ >> "$0"; exec "$@"\' "$f" "$@" <&3) & '
    "wait $!"
)

# As root, one pass over /proc: kill the live processes (zombies are already dead, and
# only wait for a reaper) that match, printing each pid killed. `uid N` matches by real
# uid (`Uid:` in /proc/<pid>/status); `sessions FILE` by a session listed in FILE (field
# 6 of /proc/<pid>/stat, counted after the command name, which may contain spaces). One
# grep or cat reads every process, skipping the ones that exit meanwhile.
_KILL_PASS = (
    'if [ "$1" = uid ]; then '
    "pids=$(grep -H -E '^(State|Uid):' /proc/[0-9]*/status 2>/dev/null | awk -F: -v uid=\"$2\" "
    '\'{split($1, p, "/"); split($3, v, " ")} $2 == "State" {st[p[3]] = v[1]} '
    '$2 == "Uid" && v[1] == uid && st[p[3]] != "Z" {print p[3]}\'); '
    'else [ -s "$2" ] || exit 0; '
    "pids=$(cat /proc/[0-9]*/stat 2>/dev/null | awk 'NR == FNR {want[$1] = 1; next} "
    '{pid = $1; sub(/^.*\\) /, ""); if ($1 != "Z" && ($4 in want)) print pid}\' "$2" -); fi; '
    'for p in $pids; do kill -9 "$p" 2>/dev/null && echo "$p"; done; exit 0'
)


async def _kill(root: Runtime, *selector: str, what: str) -> None:
    """Repeat `_KILL_PASS` until a pass finds nothing; a fork in flight can land
    between passes."""
    for _ in range(10):
        if not (await _check(root, _KILL_PASS, *selector, what=what)).strip():
            return
    raise RuntimeError(f"untrusted: {what}: processes survived being killed")


@dataclass(frozen=True, eq=False)
class UntrustedRuntime(Runtime):
    """A `Runtime` view whose commands run untrusted: as its user, each in a new session,
    with `clean_env` in place of the environment. Made by `untrusted`."""

    clean_env: Mapping[str, str] | None = None
    sessions: str = ""
    """Root-only file recording each command's session, when the view runs as root."""

    def _wrap(self, argv: list[str], env: dict[str, str]) -> list[str]:
        variables = {**(self.clean_env or {}), **env}
        for name in STARTUP_HOOKS:
            variables.pop(name, None)
        variables["PYTHONNOUSERSITE"] = "1"
        pairs = [f"{key}={value}" for key, value in variables.items()]
        return [
            "sh",
            "-c",
            _SESSION,
            "vf-untrusted",
            self.sessions,
            "env",
            "-i",
            *pairs,
            *argv,
        ]

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await super().run(self._wrap(argv, env), {})

    async def exec(self, argv: list[str], env: dict[str, str]) -> RuntimeProcess:
        return await super().exec(self._wrap(argv, env), {})

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        await super().run_background(self._wrap(argv, env), {}, log)


async def _check(runtime: Runtime, script: str, *args: str, what: str) -> str:
    result = await runtime.run(["sh", "-c", script, "vf-untrusted", *args], {})
    if result.exit_code != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[-500:]
        raise RuntimeError(f"untrusted: {what} failed - {detail}")
    return result.stdout


async def kill_user(runtime: Runtime, user: str) -> None:
    """Kill every process `user` owns in `runtime`, and check that none is left. `user`
    is an account name or `uid[:gid]`; it must not be root."""
    root = runtime.with_user("root")
    uid = user.split(":")[0]
    if not uid.isdigit():
        uid = (await _check(root, 'id -u "$1"', user, what=f"resolving {user}")).strip()
    if uid == "0":
        raise ValueError("untrusted: refusing to kill every process root owns")
    # Root sweeps /proc: `kill -9 -1` as the user would be simpler, but on Prime it
    # also kills the exec job that runs the command as that user.
    await _kill(root, "uid", uid, what=f"killing {user}'s processes")


@asynccontextmanager
async def untrusted(
    runtime: Runtime,
    task: Task,
    *,
    user: str | Literal["agent", "fresh"] = "agent",
    private: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> AsyncIterator[UntrustedRuntime]:
    """A view of `runtime` for running code the agent delivered, during grading.

    Its `run`, `exec` and `run_background` run as `user`, each in a new session, with
    `env` in place of the environment (by default PATH, LANG and HOME), with
    `PYTHONNOUSERSITE=1`, and with the startup hooks `PYTHONSTARTUP`, `BASH_ENV` and `ENV`
    unset. Its `read` reads as that user, so a link the code planted to a file it cannot
    read fails. `private` paths are made root-only (`root:root 0700`) while the block runs,
    then restored. On exit, every process the user owns is killed, and the block fails if
    any survives, before the grader reads a result.

    `user="agent"` runs as the task's `agent_user`: the delivered code keeps the
    privileges the agent had. `user="fresh"` runs as a new UID with no account, for
    tasks whose agent ran as root but whose delivered code need not; it needs a backend
    that execs as a bare UID (Docker does; Prime refuses it). Any other value names an existing user.

    When that user is root (an agent that ran as the default user), nothing here keeps the
    code from the grader's files, and only the sessions the view started are killed: a
    process that leaves its session survives. Use `"fresh"` instead where you can.

    A service the grader probes is started with `run_background` inside the block and
    probed from the trusted side; it dies when the block exits. Code the grader imports
    into its own process (pytest importing a delivered package) is not covered: no user or
    process boundary protects a process from code it loads itself.

    Needs `SUPPORTS_USER_SWITCHING`, so the subprocess runtime refuses it."""
    if not runtime.backend.SUPPORTS_USER_SWITCHING:
        raise ValueError(
            f"the {runtime.type} runtime cannot run delivered code untrusted: it "
            "cannot run as another user"
        )
    root = runtime.with_user("root")
    token = secrets.token_hex(8)
    scratch = f"/tmp/vf-untrusted-{token}"
    await _check(root, 'mkdir -m 700 "$1"', scratch, what="preparing")
    home = None
    if user == "agent":
        name, home = task.data.agent_user, task.data.agent_home
    elif user == "fresh":
        uid = 20000 + secrets.randbelow(40000)
        name = f"{uid}:{uid}"
        home = f"{scratch}-home"
        await _check(
            root,
            'mkdir -m 700 "$1" && chown "$2" "$1"',
            home,
            name,
            what="preparing a home",
        )
    else:
        name = user
    as_root = name in (None, "root", "0", "0:0")
    if home is None and not as_root:
        home = (
            await _check(
                root,
                'getent passwd "$1" | cut -d: -f6',
                str(name),
                what=f"reading {name}'s home",
            )
        ).strip() or "/"
    clean_env = (
        dict(env)
        if env is not None
        else {
            "PATH": DEFAULT_PATH,
            "LANG": "C.UTF-8",
            "HOME": home or "/root",
        }
    )
    view = UntrustedRuntime(
        backend=runtime.backend,
        env=runtime.env,
        user=None if as_root else name,
        home=None if as_root else home,
        clean_env=clean_env,
        sessions=f"{scratch}/sessions" if as_root else "",
    )
    if user == "fresh":
        try:
            await view.run(["true"], {})
        except SandboxError as exc:
            await root.run(["rm", "-rf", scratch, f"{scratch}-home"], {})
            raise ValueError(
                f"the {runtime.type} runtime cannot run as a UID with no account; "
                "use user='agent' or an existing user"
            ) from exc
    # Each private path's owner and mode, to restore: `uid:gid mode path` per line.
    saved = f"{scratch}/private"
    if private:
        await _check(
            root,
            'out=$1; shift; for p in "$@"; do [ -e "$p" ] || continue; '
            'stat -c "%u:%g %a $p" "$p" >> "$out" && chown root:root "$p" && chmod 700 "$p" '
            "|| exit 1; done",
            saved,
            *private,
            what="making paths private",
        )
    try:
        yield view
    finally:
        try:
            if as_root:
                await _kill(root, "sessions", view.sessions, what="killing sessions")
            else:
                await kill_user(runtime, str(name))
        finally:
            await _check(
                root,
                '[ -f "$1" ] && while read -r owner mode path; do chown "$owner" "$path" && '
                'chmod "$mode" "$path" || exit 1; done < "$1"; rm -rf "$2" "$2-home"',
                saved,
                scratch,
                what="restoring private paths",
            )
