"""Conformance checks for a runtime: provision it and exercise the backend contract
(`run`, `exec`, `run_background`, file I/O) and every feature it declares. Built-in and
custom runtimes alike should pass. Run them with `scripts/check_runtime.py`.

The checks expect `sh` and Python 3 in the runtime's default image."""

import asyncio
import contextlib
import re
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import aiohttp

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime, RuntimeConfig, backend_cls, provision_runtime
from verifiers.v1.runtimes.compose import INSTALL_DOCKER

PYTHON = ["sh", "-c", 'exec "$(command -v python3 || command -v python)" "$@"', "py"]
IMAGE = "python:3.12-alpine"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool | None
    """None when the runtime does not declare the feature, so the check is skipped."""
    detail: str = ""


async def wait_for(
    predicate: Callable[[], Awaitable[bool]], timeout: float = 120
) -> None:
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.5)


@contextlib.asynccontextmanager
async def provisioned(config: RuntimeConfig, **updates) -> AsyncIterator[Runtime]:
    if updates:
        config = type(config).model_validate({**config.model_dump(), **updates})
    async with provision_runtime(config) as view:
        yield view


async def check_run(config: RuntimeConfig, image: str) -> None:
    async with provisioned(config) as view:
        result = await view.run(
            ["sh", "-c", 'echo "$GREETING"; exit 3'], {"GREETING": "hi"}
        )
        assert result.exit_code == 3, result
        assert result.stdout.strip() == "hi", result
        stdin = await view.run(["cat"], {})
        assert stdin.exit_code == 0 and stdin.stdout == "", "run must close stdin"


async def check_exec(config: RuntimeConfig, image: str) -> None:
    async with provisioned(config) as view:
        process = await view.exec(
            ["sh", "-c", 'read line; echo "got $line"; echo oops >&2; exit 4'], {}
        )
        await process.write(b"ping\n")
        stdout = b"".join([chunk async for chunk in process.stdout])
        stderr = b"".join([chunk async for chunk in process.stderr])
        assert await process.wait() == 4
        assert stdout.strip() == b"got ping", stdout
        assert stderr.strip() == b"oops", stderr
        assert await process.poll() == 4
        sleeper = await view.exec(["sleep", "600"], {})
        assert await sleeper.poll() is None
        await sleeper.kill()
        async with asyncio.timeout(30):
            assert await sleeper.wait() != 0


async def check_run_background(config: RuntimeConfig, image: str) -> None:
    async with provisioned(config) as view:
        await view.run_background(["sh", "-c", "echo up; sleep 600"], {}, "bg.log")

        async def started() -> bool:
            with contextlib.suppress(SandboxError, OSError):
                return (await view.read("bg.log")).strip() == b"up"
            return False

        await wait_for(started)


async def check_files(config: RuntimeConfig, image: str) -> None:
    async with provisioned(config) as view:
        data = bytes(range(256)) * 64
        await view.write("vf-check/nested/data.bin", data)
        assert await view.read("vf-check/nested/data.bin") == data
        assert await view.read("vf-check/nested/data.bin", max_bytes=len(data)) == data
        try:
            await view.read("vf-check/nested/data.bin", max_bytes=10)
        except SandboxError:
            pass
        else:
            raise AssertionError("a read past max_bytes must raise")
        listed = await view.run(["sh", "-c", "wc -c < vf-check/nested/data.bin"], {})
        assert listed.stdout.strip() == str(len(data)), "write must land in the workdir"


async def check_task_images(config: RuntimeConfig, image: str) -> None:
    async with provisioned(config, image=image) as view:
        match = re.match(r"python:(\d+\.\d+)", image)
        if match is None:
            assert (await view.run(["true"], {})).exit_code == 0
            return
        result = await view.run([*PYTHON, "--version"], {})
        assert match.group(1) in result.stdout, result


async def check_user_switching(config: RuntimeConfig, image: str) -> None:
    async with provisioned(config) as view:
        nobody = view.with_user("nobody")
        result = await nobody.run(["id", "-un"], {})
        assert result.stdout.strip() == "nobody", result


async def check_port_exposure(config: RuntimeConfig, image: str) -> None:
    async with provisioned(config) as view:
        port = view.published_port
        if port is None:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        await view.write("vf-www/index.html", b"exposed")
        await view.run_background(
            [*PYTHON, "-m", "http.server", str(port), "-d", "vf-www"], {}, "http.log"
        )
        url = await view.expose(port)

        async def serving() -> bool:
            with contextlib.suppress(aiohttp.ClientError):
                async with (
                    aiohttp.ClientSession() as session,
                    session.get(url) as response,
                ):
                    return await response.text() == "exposed"
            return False

        await wait_for(serving)


async def check_nested_containers(config: RuntimeConfig, image: str) -> None:
    async with provisioned(config) as view:
        installed = await view.run(["sh", "-c", INSTALL_DOCKER], {})
        assert installed.exit_code == 0, installed.stdout + installed.stderr
        await view.run_background(
            ["dockerd", "--host=unix:///var/run/docker.sock"], {}, "/tmp/dockerd.log"
        )

        async def daemon_up() -> bool:
            return (await view.run(["docker", "info"], {})).exit_code == 0

        await wait_for(daemon_up)
        result = await view.run(
            ["docker", "run", "--rm", "alpine:3.20", "echo", "nested"], {}
        )
        assert result.stdout.strip() == "nested", result


async def check_network_policies(config: RuntimeConfig, image: str) -> None:
    probe = (
        "import sys, urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen(sys.argv[1], timeout=10)\n"
        "    print('ok')\n"
        "except Exception:\n"
        "    print('denied')\n"
    )

    async def reach(view: Runtime, url: str) -> str:
        return (await view.run([*PYTHON, "-c", probe, url], {})).stdout.strip()

    async with provisioned(config, allow=["pypi.org"]) as view:
        assert await reach(view, "https://example.com") == "ok", "setup must be open"
        await view.backend.prepare_execution([])
        assert await reach(view, "https://pypi.org/simple/") == "ok"
        assert await reach(view, "https://example.com") == "denied"


Check = Callable[[RuntimeConfig, str], Awaitable[None]]
CORE: dict[str, Check] = {
    "run": check_run,
    "exec": check_exec,
    "run_background": check_run_background,
    "files": check_files,
}
FEATURES: dict[str, tuple[Callable[[RuntimeConfig], bool], Check]] = {
    "task_images": (
        lambda config: backend_cls(config).SUPPORTS_TASK_IMAGES,
        check_task_images,
    ),
    "user_switching": (
        lambda config: backend_cls(config).SUPPORTS_USER_SWITCHING,
        check_user_switching,
    ),
    "port_exposure": (
        lambda config: backend_cls(config).SUPPORTS_PORT_EXPOSURE,
        check_port_exposure,
    ),
    "nested_containers": (
        lambda config: backend_cls(config).SUPPORTS_NESTED_CONTAINERS,
        check_nested_containers,
    ),
    "network_policies": (
        lambda config: isinstance(config, NetworkPolicyConfig),
        check_network_policies,
    ),
}


async def check_runtime(
    config: RuntimeConfig, *, image: str = IMAGE, timeout: float = 900
) -> list[CheckResult]:
    """Run every core check and each declared feature's check, concurrently.
    `image` is the task image the `task_images` check boots."""

    async def run(name: str, check: Check) -> CheckResult:
        try:
            async with asyncio.timeout(timeout):
                await check(config, image)
        except Exception as e:  # noqa: BLE001 - a failing check is a result, not a crash
            return CheckResult(name, False, f"{type(e).__name__}: {e}")
        return CheckResult(name, True)

    checks = dict(CORE)
    skipped = []
    for name, (declared, check) in FEATURES.items():
        if declared(config):
            checks[name] = check
        else:
            skipped.append(CheckResult(name, None, "not declared"))
    results = await asyncio.gather(*(run(n, c) for n, c in checks.items()))
    return [*results, *skipped]
