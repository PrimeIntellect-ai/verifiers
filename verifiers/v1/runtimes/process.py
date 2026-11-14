"""Private process view for model-controlled RLM execution workers."""

from __future__ import annotations

from verifiers.v1.configs.runtime import FilesystemPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.filesystem import prepare_filesystem
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.logging import clip

_INSTALL = """
set -eu
if command -v bwrap >/dev/null 2>&1; then exit 0; fi
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bubblewrap
elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache bubblewrap
elif command -v dnf >/dev/null 2>&1; then
    dnf install -y -q bubblewrap
else
    echo 'bubblewrap is not installed and no supported package manager is available' >&2
    exit 1
fi
"""


async def ensure_process_sandbox(runtime: Runtime) -> None:
    """Install the namespace launcher during trusted setup, before network policy."""
    if not runtime.backend.SUPPORTS_FILESYSTEM_POLICY:
        raise SandboxError(
            f"the {runtime.type} runtime cannot isolate RLM execution processes"
        )
    result = await runtime.with_user("root").run(["/bin/sh", "-c", _INSTALL], {})
    if result.exit_code:
        raise SandboxError(
            "cannot install bubblewrap for RLM process isolation - "
            + (clip(result.stderr or result.stdout) or "unknown error")
        )


async def execution_command(
    runtime: Runtime,
    install_dir: str,
    rlm_home: str,
    policy: FilesystemPolicyConfig | None,
) -> tuple[list[str], str]:
    """Return an RLM worker command, refusing an unverified namespace boundary."""
    python = f"{install_dir}/tools/rlm/bin/python"
    outer = await runtime.run(
        [
            "/bin/sh",
            "-c",
            'pwd -P; printf "%s\\n" "$HOME"; readlink /proc/self/ns/pid; test -x "$1"',
            "process-probe",
            python,
        ],
        {},
    )
    fields = outer.stdout.splitlines()
    if outer.exit_code or len(fields) != 3:
        raise SandboxError(
            "cannot prepare RLM process isolation - "
            + (clip(outer.stderr or outer.stdout) or "unknown error")
        )
    cwd, user_home, outer_pid_namespace = fields
    if policy is not None:
        prefix = list((await prepare_filesystem(runtime, policy)).prefix)
    else:
        probe = await runtime.run(
            [
                "/bin/sh",
                "-c",
                'readlink -f "$(command -v bwrap)"',
            ],
            {},
        )
        binary = probe.stdout.strip()
        if probe.exit_code or not binary:
            raise SandboxError(
                "cannot find bubblewrap for RLM process isolation - "
                + (clip(probe.stderr or probe.stdout) or "unknown error")
            )
        if not binary.startswith(("/usr/", "/bin/", "/sbin/")):
            raise SandboxError(
                "bubblewrap must be installed in a protected system directory"
            )
        system_root = "/" + binary.strip("/").split("/", 1)[0]
        prefix = [
            binary,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--disable-userns",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--cap-drop",
            "ALL",
            "--bind",
            "/",
            "/",
            "--ro-bind",
            system_root,
            system_root,
            "--ro-bind",
            "/etc",
            "/etc",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--chdir",
            cwd,
        ]
    # The supervisor's session directory may contain its ledger and metadata.
    # Give the worker an empty session tree while retaining the skill packages
    # installed under RLM_HOME/skills. Keep the editable RLM install read-only so
    # a cell cannot alter code the credential-bearing supervisor may import.
    # Clear inherited model/tool credentials.
    path = ":".join(
        (
            f"{install_dir}/bin",
            f"{install_dir}/tools/rlm/bin",
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        )
    )
    command = [
        *prefix,
        "--ro-bind",
        install_dir,
        install_dir,
        "--tmpfs",
        f"{rlm_home}/sessions",
        "--clearenv",
        "--setenv",
        "HOME",
        user_home,
        "--setenv",
        "RLM_HOME",
        rlm_home,
        "--setenv",
        "PATH",
        path,
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "TMPDIR",
        "/tmp",
        "--",
        python,
        "-m",
        "rlm.execution",
    ]
    check = await runtime.run(
        [*command[:-3], "/bin/sh", "-c", "readlink /proc/self/ns/pid"],
        {},
    )
    if check.exit_code or check.stdout.strip() == outer_pid_namespace:
        raise SandboxError(
            "RLM process isolation requires working bubblewrap user/PID namespaces "
            "and a private /proc - "
            + (clip(check.stderr or check.stdout) or "unknown error")
        )
    return command, cwd
