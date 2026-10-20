"""The image's `ENTRYPOINT`, started as Harbor's Docker environment starts it.

Harbor runs a task's container with the image's `ENTRYPOINT` and its own command,
`sh -c "sleep infinity"`, in place of the image's `CMD`. Images rely on that start:
one copies a starter tree into the workdir, one starts the database its app serves,
one drops `sleep` to the agent's user. A provider box whose first process is its
own init never runs it, so the task's setup starts it — once the box is up, before
anything else the task sets up — and waits until it reaches the command it was
handed, or execs into something else (`exec sleep infinity` after starting services,
a server in the foreground), or exits.

The entrypoint is read from the Dockerfile the task ships for the image (its final
stage, following `FROM <stage>`); an entrypoint a base image declares is not known.
"""

import asyncio
import json
import logging
import re
import shlex
import time
from pathlib import Path, PurePosixPath

from verifiers.v1.runtimes import Runtime

logger = logging.getLogger(__name__)

READY = "vf-harbor-entrypoint-ready"
"""Carried on the stand-in command's argv: its process shows the entrypoint got there."""
COMMAND = ["sh", "-c", "while :; do sleep 3600; done", READY]
"""Harbor's `sleep infinity`, named so it can be found once the entrypoint reaches it."""
STATE = "/run/vf-harbor-entrypoint"
LOG = f"{STATE}.log"
PID = f"{STATE}.pid"
RECORD = 'echo $$ > {pid} || exit 1; exec "$@"'
LAUNCH = (
    "record=$1; shift; command -v setsid >/dev/null 2>&1 && "
    'exec setsid sh -c "$record" "$0" "$@"; exec sh -c "$record" "$0" "$@"'
)
"""Records the entrypoint's pid, then runs it in a session of its own, as a container's
first process is: a box's command daemon shares the process group of what it starts,
and the platform drops a box whose group holds a stopped process (a grader that
`pkill -STOP`s the agent's user would stop the entrypoint's `runuser` child)."""
WAIT_SECONDS = 600.0
FOREGROUND_GRACE_SECONDS = 5.0
OUTPUT_BYTES = 4000
SHELLS = {"sh", "bash", "dash", "ash", "zsh", "ksh", "busybox"}
"""A process still running one of these is the entrypoint's script, not yet exec'd."""

HEREDOC = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")


def instructions(text: str) -> list[tuple[str, str]]:
    """A Dockerfile's instructions as (keyword, arguments): continuations are joined,
    comments dropped and a heredoc's body skipped."""
    escape = "`" if re.match(r"(?i)^\s*#\s*escape\s*=\s*`", text) else "\\"
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        joined = [line.rstrip()]
        while joined[-1].endswith(escape) and i + 1 < len(lines):
            joined[-1] = joined[-1][:-1]
            i += 1
            following = lines[i]
            if following.lstrip().startswith("#"):
                joined.append(escape)  # a comment inside a continuation is dropped
                continue
            joined.append(following.rstrip())
        if joined[-1] == escape:
            joined[-1] = ""
        body = " ".join(part.strip() for part in joined)
        keyword, _, arguments = body.strip().partition(" ")
        out.append((keyword.upper(), arguments.strip()))
        i += 1
        for delimiter in HEREDOC.findall(body):
            while i < len(lines) and lines[i].strip() != delimiter:
                i += 1
            i += 1
    return out


def exec_form(arguments: str) -> list[str] | None:
    """An instruction's JSON (exec) form, or None for its shell form."""
    if not arguments.startswith("["):
        return None
    try:
        value = json.loads(arguments)
    except ValueError:
        return None
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None


def image_start(dockerfile: Path) -> tuple[list[str] | None, bool, str | None]:
    """The final stage's entrypoint argv, whether it is a shell form (which takes no
    command), and its `USER`. A stage built `FROM` an earlier one inherits both; an
    entrypoint inherited from an outside base image is unknown and reads as none."""
    Start = tuple[list[str] | None, bool, str | None]
    stages: dict[str, Start] = {}
    entrypoint, shell, user, alias = None, False, None, None
    variables: dict[str, str] = {}

    def expand(value: str) -> str:
        return re.sub(
            r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)",
            lambda m: variables.get(m[1] or m[2], m[0]),
            value,
        )

    for keyword, arguments in instructions(dockerfile.read_text(errors="replace")):
        if keyword == "FROM":
            if alias is not None:
                stages[alias] = (entrypoint, shell, user)
            tokens = [t for t in arguments.split() if not t.startswith("--")]
            base = tokens[0].lower() if tokens else ""
            entrypoint, shell, user = stages.get(base, (None, False, None))
            has_alias = len(tokens) >= 3 and tokens[1].upper() == "AS"
            alias = tokens[2].lower() if has_alias else None
        elif keyword in ("ARG", "ENV"):
            try:
                values = shlex.split(arguments)
            except ValueError:
                continue
            if keyword == "ENV" and values and "=" not in values[0]:
                variables[values[0]] = " ".join(values[1:])
                continue
            for item in values:
                name, sep, value = item.partition("=")
                if sep:
                    variables[name] = expand(value)
        elif keyword == "ENTRYPOINT":
            argv = exec_form(arguments)
            shell = argv is None
            entrypoint = ["/bin/sh", "-c", arguments] if shell else argv or None
        elif keyword == "USER":
            named = expand(arguments.split()[0]) if arguments.split() else ""
            user = None if "$" in named else named
    return entrypoint, shell, user


def entrypoint_argv(dockerfile: Path) -> tuple[list[str], str | None] | None:
    """The argv Harbor's Docker start runs for this image (entrypoint plus Harbor's
    command, which a shell-form entrypoint ignores), and the user it runs as; None
    when the image declares no entrypoint."""
    if not dockerfile.is_file():
        return None
    entrypoint, shell, user = image_start(dockerfile)
    if not entrypoint:
        return None
    if any(PurePosixPath(arg).parts[-2:] == ("tests", "test.sh") for arg in entrypoint):
        # A verifier image built to run its tests as a one-shot container: its start is
        # the grade itself, which Harbor runs once /tests and the artifacts are staged.
        return None
    # A shell form takes no command: the marker becomes its `$0`, which nothing reads
    # but its error messages.
    argv = [*entrypoint, READY] if shell else [*entrypoint, *COMMAND]
    user = user.split(":")[0] if user else None
    return argv, None if user in (None, "", "root", "0") else user


WATCH = r"""
pid=$(cat "$1" 2>/dev/null) || exit 3
[ -n "$pid" ] || exit 3
for p in /proc/[0-9]*; do
  tr '\0' ' ' < "$p/cmdline" 2>/dev/null | grep -qF "$2" || continue
  [ "$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)" = "$3 " ] && { echo reached; exit 0; }
done
[ -d "/proc/$pid" ] || { echo exited; exit 0; }
cmd=$(tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null) || { echo exited; exit 0; }
[ -n "$cmd" ] || { echo exited; exit 0; }
printf '%s\n' "$cmd" | grep -qxF "$2" || { echo execd; exit 0; }
echo "running $(basename "$(printf '%s\n' "$cmd" | head -n 1)")"
"""


async def start_entrypoint(
    runtime: Runtime, dockerfile: Path, wait: float = WAIT_SECONDS
) -> dict | None:
    """Start the image's entrypoint in the background and wait until it is ready;
    returns what it did for the trace, or None when the image declares none.

    Never fails setup itself: the box is what the image's start left, as under Docker,
    and an entrypoint that exits or keeps a shell running past `wait` is recorded."""
    start = entrypoint_argv(dockerfile)
    if start is None or not runtime.supports_live_processes:
        return None
    argv, user = start
    # Root's files, made before the entrypoint runs (it may tighten /tmp, and the
    # agent must not read what it printed); the user it runs as may write them.
    owner = f"chown {shlex.quote(user)} {PID} {LOG} && " if user is not None else ""
    made = await runtime.run(
        [
            "sh",
            "-c",
            (
                f"rm -f {PID} && install -m 600 /dev/null {PID} && "
                f"install -m 600 /dev/null {LOG} && {owner}true"
            ),
        ],
        {},
    )
    if made.exit_code:
        logger.warning("entrypoint state files: %s", made.stderr.strip()[-500:])
        return None
    # The image's USER runs its entrypoint, as Docker starts it.
    starter = runtime if user is None else runtime.with_user(user)
    await starter.run_background(
        ["sh", "-c", LAUNCH, "vf-harbor-entrypoint", RECORD.format(pid=PID), *argv],
        {},
        LOG,
    )
    began = time.monotonic()
    command = " ".join(COMMAND)
    state, foreground_since = "starting", None
    while time.monotonic() - began < wait:
        probe = await runtime.run(["sh", "-c", WATCH, "vf", PID, READY, command], {})
        state = probe.stdout.strip() or "starting"
        if state in ("reached", "exited", "execd"):
            break
        if state.startswith("running ") and state.split()[1] not in SHELLS:
            # A program in the foreground (a server the image runs as its entrypoint,
            # or one a wrapper starts): Docker waits for nothing; give it a moment.
            foreground_since = foreground_since or time.monotonic()
            if time.monotonic() - foreground_since >= FOREGROUND_GRACE_SECONDS:
                state = "foreground"
                break
        await asyncio.sleep(0.5)
    else:
        logger.warning(
            "%s: entrypoint still running its script after %.0fs; continuing",
            dockerfile.parent.parent.name,
            wait,
        )
    log = await runtime.run(["sh", "-c", f"tail -c {OUTPUT_BYTES} {LOG}"], {})
    return {
        "argv": argv,
        "user": user,
        "state": state,
        "seconds": round(time.monotonic() - began, 1),
        "output": log.stdout,
    }
