"""A private Docker daemon with bounded storage inside an owned Prime VM."""

import asyncio

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import Runtime


async def prepare_docker_host(host: Runtime, disk_gb: float) -> None:
    """Reserve a loop-backed filesystem; task layers and volumes cannot fill root.

    The owning VM is disposable and supplies teardown, including after partial
    setup. The socket is available only to trusted host processes.
    """
    size = int(disk_gb * 1024**3)
    if size < 1024**3:
        raise ValueError("execution storage must be at least 1 GiB, including images")
    result = await host.run(
        [
            "sh",
            "-c",
            (
                "command -v dockerd >/dev/null && command -v mkfs.ext4 >/dev/null || "
                "{ apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io e2fsprogs; }"
            ),
        ],
        {},
    )
    if result.exit_code:
        raise SandboxError(f"Docker host setup failed: {result.stderr[-1000:]}")
    result = await host.run(["df", "-Pk", "/"], {})
    try:
        available = int(result.stdout.splitlines()[-1].split()[3]) * 1024
    except (IndexError, ValueError) as error:
        raise SandboxError(
            "cannot determine storage available for task isolation"
        ) from error
    if available - size < 1024**3:
        raise SandboxError(
            "execution storage must leave at least 1 GiB free for the harness"
        )
    result = await host.run(
        [
            "sh",
            "-c",
            (
                "set -eu; mkdir -p /var/lib/vf-execution; "
                'fallocate -l "$1" /var/lib/vf-execution.img; '
                "mkfs.ext4 -q -F -E nodiscard /var/lib/vf-execution.img; "
                "mount -o loop,nodev,nosuid /var/lib/vf-execution.img /var/lib/vf-execution; "
                "mkdir -p /var/lib/vf-execution/tmp"
            ),
            "vf-storage",
            str(size),
        ],
        {},
    )
    if result.exit_code:
        raise SandboxError(
            f"cannot enforce execution storage budget: {result.stderr[-1000:]}"
        )
    socket = "unix:///run/vf-execution.sock"
    host.env = {**host.env, "DOCKER_HOST": socket}
    await host.run_background(
        [
            "dockerd",
            "--host=" + socket,
            "--data-root=/var/lib/vf-execution/docker",
            "--exec-root=/run/vf-execution",
            "--pidfile=/run/vf-execution.pid",
            "--log-driver=local",
            "--log-opt=max-size=10m",
            "--log-opt=max-file=2",
        ],
        {"DOCKER_TMPDIR": "/var/lib/vf-execution/tmp"},
        "/var/lib/vf-execution/dockerd.log",
    )
    for _ in range(60):
        result = await host.run(["docker", "info"], {})
        if not result.exit_code:
            return
        await asyncio.sleep(1)
    raise SandboxError(
        "private Docker daemon did not start; see /var/lib/vf-execution/dockerd.log"
    )
