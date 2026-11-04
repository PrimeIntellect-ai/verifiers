#!/usr/bin/python3 -I
"""Root-only cgroup-v2 workload controller, installed inside a task runtime."""

import argparse
import fcntl
import json
import os
import pwd
import sys
import time
import uuid
from pathlib import Path

STATE = Path("/run/vf-task-scope")
PARENT = Path("/sys/fs/cgroup/vf-task-scopes")
ROLES = ("agent", "scorer")
WAIT_SECONDS = 10


def secure_path(path):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise RuntimeError("scope path is not canonical")
    for part in (path, *path.parents):
        info = part.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError("scope path is not root protected")


def events(path):
    return dict(
        line.split() for line in (path / "cgroup.events").read_text().splitlines()
    )


def wait_event(path, key, value):
    deadline = time.monotonic() + WAIT_SECONDS
    while events(path).get(key) != value:
        if time.monotonic() >= deadline:
            raise RuntimeError("scope state transition was not confirmed")
        time.sleep(0.01)


def cgroup2_mount(path):
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = Path(left.split()[4])
        if path == mount or mount in path.parents:
            mounts.append((len(mount.parts), right.split()[0]))
    return bool(mounts) and max(mounts)[1] == "cgroup2"


def prepare():
    secure_path(STATE)
    secure_path(PARENT.parent)
    if not cgroup2_mount(PARENT.parent):
        raise RuntimeError("scope parent must be on a cgroup-v2 filesystem")
    # Only Prime VM runtimes install this controller; never alter existing permissions.
    if not PARENT.exists():
        PARENT.mkdir(mode=0o700)
    secure_path(PARENT)
    if not cgroup2_mount(PARENT):
        raise RuntimeError("scope parent must be on a cgroup-v2 filesystem")
    if not (PARENT / "cgroup.controllers").is_file():
        raise RuntimeError("cgroup-v2 delegation is required")
    if not Path("/usr/bin/setpriv").is_file() or not Path("/usr/bin/env").is_file():
        raise RuntimeError("scope launch tools are unavailable")
    root = PARENT / ("vf-" + uuid.uuid4().hex)
    root.mkdir(mode=0o700)
    for role in ROLES:
        leaf = root / role
        leaf.mkdir(mode=0o700)
        for name in ("cgroup.procs", "cgroup.freeze", "cgroup.kill", "cgroup.events"):
            secure_path(leaf / name)
        if events(leaf).get("populated") != "0":
            raise RuntimeError("new scope is not empty")
    (STATE / "scope.json").write_text(json.dumps({"root": str(root)}))
    for role in ROLES:
        (STATE / (role + ".state")).write_text("active")
    (STATE / "lock").touch(mode=0o600)


def scope_path(role):
    if role not in ROLES:
        raise ValueError("unknown workload role")
    root = Path(json.loads((STATE / "scope.json").read_text())["root"])
    if root.parent != PARENT or not root.name.startswith("vf-"):
        raise RuntimeError("invalid scope registration")
    leaf = root / role
    secure_path(leaf)
    return leaf


def transition(operation, role):
    path = scope_path(role)
    if str(os.getpid()) in (path / "cgroup.procs").read_text().split():
        raise RuntimeError("scope controller must remain outside the workload")
    state_file = STATE / (role + ".state")
    state = state_file.read_text()
    if operation == "freeze":
        if state == "closed":
            raise RuntimeError("agent scope is closed")
        state_file.write_text("frozen")
        (path / "cgroup.freeze").write_text("1")
        wait_event(path, "frozen", "1")
    elif operation == "thaw":
        (path / "cgroup.freeze").write_text("0")
        wait_event(path, "frozen", "0")
        if state != "closed":
            state_file.write_text("quiescing")
    elif operation == "kill":
        # Scorer resets serialize with enrollment; the agent never restarts after kill.
        state_file.write_text("closed")
        (path / "cgroup.kill").write_text("1")
        wait_event(path, "populated", "0")
        if role == "scorer":
            (path / "cgroup.freeze").write_text("0")
            wait_event(path, "frozen", "0")
            state_file.write_text("active")


def launch(role, argv, user=None):
    if not argv:
        raise ValueError("workload command is required")
    if (STATE / (role + ".state")).read_text() != "active":
        raise RuntimeError("workload scope is not accepting launches")
    path = scope_path(role)
    if str(os.getpid()) in (path / "cgroup.procs").read_text().split():
        raise RuntimeError("scope controller must remain outside the workload")
    child_env = {
        key: value for key, value in os.environ.items() if key != "VF_SCOPE_CHILD_ENV"
    }
    overrides = json.loads(os.environ.get("VF_SCOPE_CHILD_ENV", "null"))
    if overrides is not None:
        child_env.update(overrides)
    if user is not None:
        account = pwd.getpwnam(user)
        if account.pw_uid == 0:
            raise RuntimeError("agent workload requires a non-root account")
        argv = [
            "/usr/bin/setpriv",
            "--reuid",
            str(account.pw_uid),
            "--regid",
            str(account.pw_gid),
            "--init-groups",
            "--no-new-privs",
            "--bounding-set=-all",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--",
            "/usr/bin/env",
            "-i",
            *(f"{key}={value}" for key, value in child_env.items()),
            *argv,
        ]
        child_env = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"}
    (path / "cgroup.procs").write_text(str(os.getpid()))
    return argv, child_env


def main():
    if os.geteuid() != 0:
        raise RuntimeError("scope controller requires root")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("prepare")
    for operation in ("freeze", "thaw", "kill"):
        item = sub.add_parser(operation)
        item.add_argument("role", choices=ROLES)
    for operation in ("run", "exec"):
        item = sub.add_parser(operation)
        item.add_argument("role", choices=ROLES)
        if operation == "run":
            item.add_argument("--user", required=True)
    raw = sys.argv[1:]
    separator = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:separator])
    command = raw[separator + 1 :]
    secure_path(STATE)
    if args.operation == "prepare":
        prepare()
        return
    with (STATE / "lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.operation in ("run", "exec"):
            argv = command
            if args.operation == "exec" and args.role != "scorer":
                raise RuntimeError("trusted exec is only available for scorers")
            argv, env = launch(args.role, argv, getattr(args, "user", None))
        else:
            transition(args.operation, args.role)
            return
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - payload-free CLI failure boundary
        # Never reproduce workload commands, environment, or provider output.
        print("workload scope failed: " + type(error).__name__, file=sys.stderr)
        sys.exit(1)
