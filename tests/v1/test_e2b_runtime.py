import asyncio
import base64
import sys
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import TypeAdapter

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import (
    RuntimeConfig,
    RuntimeInfo,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.runtimes.e2b import (
    E2BConfig,
    E2BProcess,
    E2BRuntime,
    E2BRuntimeInfo,
    _network_selector,
)


class FakeCommandExitException(Exception):
    def __init__(
        self,
        exit_code: int,
        *,
        stdout: str = "",
        stderr: str = "",
        error: str | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.error = error
        super().__init__(error or f"exit {exit_code}")


class FakeFiles:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.made: list[str] = []

    async def make_dir(self, path: str) -> bool:
        self.made.append(path)
        return True

    async def write(self, path: str, data: bytes) -> None:
        self.data[path] = data

    async def read(self, path: str, format: str = "text"):
        data = self.data[path]
        if format == "bytes":
            return bytearray(data)
        return data.decode()


class FakeHandle:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.pid = 123
        self.result = result or SimpleNamespace(exit_code=0, stdout="", stderr="")
        self.error = error
        self.stdin: list[bytes] = []
        self.killed = False
        self.wait_event: asyncio.Event | None = None

    async def send_stdin(self, data: bytes) -> None:
        self.stdin.append(data)

    async def wait(self):
        if self.wait_event is not None:
            await self.wait_event.wait()
        if self.error is not None:
            raise self.error
        return self.result

    async def kill(self) -> bool:
        self.killed = True
        return True


class FakeCommands:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.next_result = SimpleNamespace(exit_code=0, stdout="ok", stderr="")
        self.next_error: Exception | None = None
        self.handle = FakeHandle()
        self.start_entered: asyncio.Event | None = None
        self.start_release: asyncio.Event | None = None
        self.handle_wait_event: asyncio.Event | None = None

    async def run(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        if kwargs.get("background"):
            callback = kwargs.get("on_stdout")
            if callback is not None:
                if "base64" in command:  # a live process: framed bytes, split mid-line
                    wire = base64.b64encode(b"early output").decode() + "\n"
                    wire += base64.b64encode(b"\x00\xbf\xff").decode() + "\n"
                    callback(wire[:7])
                    callback(wire[7:])
                else:
                    callback("early output")
            if self.next_error is not None:
                error, self.next_error = self.next_error, None
                self.handle = FakeHandle(error=error)
            else:
                result = self.next_result
                if "base64" in command:
                    result = SimpleNamespace(
                        exit_code=0,
                        stdout=base64.b64encode(b"abcd").decode(),
                        stderr="",
                    )
                self.handle = FakeHandle(result=result)
            self.handle.wait_event = self.handle_wait_event
            if self.start_entered is not None:
                self.start_entered.set()
            if self.start_release is not None:
                await self.start_release.wait()
            return self.handle
        return self.next_result


class FakeSandbox:
    created: ClassVar[list[tuple[tuple, dict]]] = []
    next: "FakeSandbox | None" = None

    def __init__(self, sandbox_id: str = "sandbox-1") -> None:
        self.sandbox_id = sandbox_id
        self.files = FakeFiles()
        self.commands = FakeCommands()
        self.policies: list[dict] = []
        self.killed = False

    @classmethod
    async def create(cls, *args, **kwargs):
        cls.created.append((args, kwargs))
        sandbox = cls.next or cls()
        cls.next = None
        return sandbox

    async def update_network(self, policy: dict) -> None:
        self.policies.append(policy)

    def get_host(self, port: int) -> str:
        return f"{port}-{self.sandbox_id}.e2b.app"

    async def kill(self) -> bool:
        self.killed = True
        return True


class FakeAsyncTemplate:
    aliases: ClassVar[set[str]] = set()
    builds: ClassVar[list[tuple["FakeAsyncTemplate", str, dict]]] = []

    def __init__(self) -> None:
        self.image: str | None = None

    def from_image(self, image: str):
        self.image = image
        return self

    @classmethod
    async def alias_exists(cls, alias: str) -> bool:
        return alias in cls.aliases

    @classmethod
    async def build(cls, builder, name: str, **kwargs) -> None:
        cls.builds.append((builder, name, kwargs))
        cls.aliases.add(name)


class FakeSyncSandbox:
    killed: ClassVar[list[str]] = []

    @classmethod
    def kill(cls, sandbox_id: str) -> bool:
        cls.killed.append(sandbox_id)
        return True


@pytest.fixture
def fake_e2b(monkeypatch):
    FakeSandbox.created = []
    FakeSandbox.next = None
    FakeAsyncTemplate.aliases = set()
    FakeAsyncTemplate.builds = []
    FakeSyncSandbox.killed = []
    module = SimpleNamespace(
        AsyncSandbox=FakeSandbox,
        AsyncTemplate=FakeAsyncTemplate,
        Sandbox=FakeSyncSandbox,
        CommandExitException=FakeCommandExitException,
    )
    monkeypatch.setitem(sys.modules, "e2b", module)
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    return module


def test_config_parsing_and_network_validation():
    config = TypeAdapter(RuntimeConfig).validate_python(
        {"type": "e2b", "allow": ["api.example.com", "*.github.com"]}
    )
    assert isinstance(config, E2BConfig)
    assert not runtime_is_local(config)
    assert _network_selector("10.0.0.1") == "10.0.0.1"
    assert _network_selector("10.0.0.0/8") == "10.0.0.0/8"
    assert _network_selector("http://localhost:8000", framework=True) is None
    with pytest.raises(ValueError, match="would be broadened"):
        E2BConfig(allow=["https://example.com"])
    with pytest.raises(ValueError, match="would be broadened"):
        E2BConfig(allow=["example.com:8080"])
    with pytest.raises(ValueError, match="cannot deny a DNS name"):
        E2BConfig(block=["example.com"])
    with pytest.raises(ValueError, match="invalid E2B egress rule"):
        E2BConfig(allow=["not a cidr or host!"])


def test_registry_and_info_round_trip(fake_e2b):
    config = E2BConfig(template="base", timeout=600)
    runtime = make_runtime(config, "vf-test")
    assert isinstance(runtime, E2BRuntime)
    info = TypeAdapter(RuntimeInfo).validate_python(
        {
            **config.model_dump(),
            "id": "sandbox-1",
            "borrowed": False,
            "resolved_template": "base",
        }
    )
    assert isinstance(info, E2BRuntimeInfo)
    assert TypeAdapter(RuntimeInfo).validate_python(info.model_dump()) == info


def test_runtime_requires_auth(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    with pytest.raises(SystemExit, match=r"set \$E2B_API_KEY"):
        E2BRuntime(E2BConfig())


@pytest.mark.asyncio
async def test_start_run_and_teardown(fake_e2b):
    sandbox = FakeSandbox()
    FakeSandbox.next = sandbox
    runtime = E2BRuntime(E2BConfig(template="existing", timeout=123), name="vf-test")
    runtime.env = {"BASE": "1"}

    await runtime.start()
    assert runtime.info.id == "sandbox-1"
    assert runtime.info.resolved_template == "existing"
    assert FakeSandbox.created == [
        (
            ("existing",),
            {
                "timeout": 123,
                "envs": {"BASE": "1"},
                "metadata": {"verifiers-runtime": "vf-test"},
            },
        )
    ]
    assert sandbox.files.made == ["/home/user"]

    result = await runtime.run(["printf", "%s", "a b"], {"CHILD": "2"})
    assert result.exit_code == 0
    command, kwargs = sandbox.commands.calls[-1]
    assert command == "printf %s 'a b'"
    assert kwargs["envs"] == {"BASE": "1", "CHILD": "2"}
    assert kwargs["cwd"] == "/home/user"
    assert kwargs["timeout"] == 0

    sandbox.commands.next_error = FakeCommandExitException(
        42, stdout="out", stderr="err", error="exit status 42"
    )
    result = await runtime.run(["false"], {})
    assert (result.exit_code, result.stdout, result.stderr) == (42, "out", "err")

    await runtime.stop()
    assert sandbox.killed
    assert runtime.stopped


@pytest.mark.asyncio
async def test_image_build_is_deterministic_and_cached(fake_e2b):
    first = E2BRuntime(E2BConfig(image="python:3.11-slim"))
    second = E2BRuntime(E2BConfig(image="python:3.11-slim"))
    name = await first._resolve_template()
    assert name.startswith("vf-")
    assert len(FakeAsyncTemplate.builds) == 1
    builder, built_name, kwargs = FakeAsyncTemplate.builds[0]
    assert built_name == name
    assert builder.image == "python:3.11-slim"
    assert kwargs == {}
    assert await second._resolve_template() == name
    assert len(FakeAsyncTemplate.builds) == 1


@pytest.mark.asyncio
async def test_image_build_reuses_alias_after_concurrent_build(fake_e2b, monkeypatch):
    async def competing_build(builder, name: str, **kwargs) -> None:
        FakeAsyncTemplate.aliases.add(name)
        raise RuntimeError("alias already exists")

    monkeypatch.setattr(FakeAsyncTemplate, "build", staticmethod(competing_build))
    runtime = E2BRuntime(E2BConfig(image="python:3.12-slim"))
    assert (await runtime._resolve_template()).startswith("vf-")


@pytest.mark.asyncio
async def test_files_expose_and_network_policy(fake_e2b):
    runtime = E2BRuntime(E2BConfig(allow=["1.1.1.1"], workdir="/workspace"))
    sandbox = FakeSandbox()
    runtime._sandbox = sandbox
    runtime.info.id = sandbox.sandbox_id

    await runtime.write("nested/file", b"abcdef")
    assert sandbox.files.data["/workspace/nested/file"] == b"abcdef"
    assert await runtime._read("nested/file") == b"abcdef"
    assert await runtime._read("nested/file", max_bytes=4) == b"abcd"
    assert "head -c" in sandbox.commands.calls[-1][0]
    assert await runtime.expose(8123) == "https://8123-sandbox-1.e2b.app"
    with pytest.raises(ValueError, match="between 1 and 65535"):
        await runtime.expose(0)

    await runtime.prepare_execution(
        ["https://framework.example.com/v1", "http://127.0.0.1:9000"]
    )
    assert sandbox.policies[-1] == {
        "allow_out": ["framework.example.com", "1.1.1.1"],
        "deny_out": ["0.0.0.0/0"],
    }
    await runtime.prepare_execution(None)
    assert sandbox.policies[-1] == {"allow_internet_access": True}


@pytest.mark.asyncio
async def test_ip_only_policy_adds_dns_resolver(fake_e2b):
    runtime = E2BRuntime(E2BConfig(allow=["1.1.1.1"]))
    sandbox = FakeSandbox()
    runtime._sandbox = sandbox
    await runtime.prepare_execution(["https://2.2.2.2"])
    assert sandbox.policies[-1] == {
        "allow_out": ["2.2.2.2", "1.1.1.1", "8.8.8.8"],
        "deny_out": ["0.0.0.0/0"],
    }


@pytest.mark.asyncio
async def test_open_process_streams_and_signals(fake_e2b):
    runtime = E2BRuntime(E2BConfig())
    sandbox = FakeSandbox()
    runtime._sandbox = sandbox

    process = await runtime.open_process(["cat"], {})
    command, kwargs = sandbox.commands.calls[-1]
    assert command.startswith("set -o pipefail; ( ") and "base64" in command
    assert await anext(process.stdout) == b"early output"
    assert await anext(process.stdout) == b"\x00\xbf\xff"  # not UTF-8, arrives intact
    await process.write(b"hello\n")
    assert sandbox.commands.handle.stdin == [b"hello\n"]
    assert await process.wait() == 0
    with pytest.raises(StopAsyncIteration):
        await anext(process.stdout)

    process = E2BProcess(
        FakeHandle(error=FakeCommandExitException(-1, error="signal: terminated")),
        sandbox.commands,
        asyncio.Queue(),
        asyncio.Queue(),
    )
    assert await process.wait() == -15

    handle = FakeHandle()
    handle.wait_event = asyncio.Event()
    process = E2BProcess(handle, sandbox.commands, asyncio.Queue(), asyncio.Queue())
    await process.terminate()
    command, kwargs = sandbox.commands.calls[-1]
    assert command == "kill -TERM 123"
    assert kwargs["timeout"] == 0
    handle.wait_event.set()
    assert await process.wait() == 0


@pytest.mark.asyncio
async def test_background_and_cleanup(fake_e2b):
    runtime = E2BRuntime(E2BConfig())
    sandbox = FakeSandbox()
    runtime._sandbox = sandbox
    runtime.info.id = sandbox.sandbox_id

    await runtime.run_background(["python", "-m", "server"], {"X": "1"}, "tool.log")
    command, kwargs = sandbox.commands.calls[-1]
    assert "nohup python -m server > tool.log 2>&1 < /dev/null &" in command
    assert kwargs["envs"] == {"PYTHONUNBUFFERED": "1", "X": "1"}

    runtime.cleanup()
    assert FakeSyncSandbox.killed == ["sandbox-1"]


@pytest.mark.asyncio
async def test_process_wait_failure_is_wrapped(fake_e2b):
    process = E2BProcess(
        FakeHandle(error=RuntimeError("transport gone")),
        FakeCommands(),
        asyncio.Queue(),
        asyncio.Queue(),
    )
    with pytest.raises(SandboxError, match="transport gone"):
        await process.wait()


@pytest.mark.asyncio
async def test_path_is_applied_after_login_shell(fake_e2b):
    runtime = E2BRuntime(E2BConfig())
    runtime.env = {"PATH": "/custom/bin", "BASE": "1"}
    sandbox = FakeSandbox()
    runtime._sandbox = sandbox

    await runtime.run(["echo", "ok"], {})
    command, kwargs = sandbox.commands.calls[-1]
    assert command == "export PATH=/custom/bin; exec echo ok"
    assert kwargs["envs"] == {"BASE": "1"}


@pytest.mark.asyncio
async def test_wildcard_policy_includes_apex(fake_e2b):
    runtime = E2BRuntime(E2BConfig(allow=["*.example.com"]))
    sandbox = FakeSandbox()
    runtime._sandbox = sandbox
    await runtime.prepare_execution(["https://framework.example.net"])
    assert sandbox.policies[-1] == {
        "allow_out": [
            "framework.example.net",
            "*.example.com",
            "example.com",
        ],
        "deny_out": ["0.0.0.0/0"],
    }


@pytest.mark.asyncio
async def test_process_start_cancellation_kills_captured_handle(fake_e2b):
    runtime = E2BRuntime(E2BConfig())
    sandbox = FakeSandbox()
    runtime._sandbox = sandbox
    sandbox.commands.start_entered = asyncio.Event()
    sandbox.commands.start_release = asyncio.Event()

    task = asyncio.create_task(runtime.open_process(["cat"], {}))
    await sandbox.commands.start_entered.wait()
    task.cancel()
    sandbox.commands.start_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.commands.handle.killed


@pytest.mark.asyncio
async def test_run_cancellation_kills_command(fake_e2b):
    runtime = E2BRuntime(E2BConfig())
    sandbox = FakeSandbox()
    runtime._sandbox = sandbox
    sandbox.commands.handle_wait_event = asyncio.Event()

    task = asyncio.create_task(runtime.run(["sleep", "10"], {}))
    while not sandbox.commands.calls:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.commands.handle.killed


@pytest.mark.asyncio
async def test_cancelled_wait_does_not_cancel_process_watcher(fake_e2b):
    handle = FakeHandle()
    handle.wait_event = asyncio.Event()
    process = E2BProcess(handle, FakeCommands(), asyncio.Queue(), asyncio.Queue())

    first_wait = asyncio.create_task(process.wait())
    await asyncio.sleep(0)
    first_wait.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_wait
    handle.wait_event.set()
    assert await process.wait() == 0


@pytest.mark.asyncio
async def test_public_read_rejects_overflow_at_source_cap(fake_e2b):
    runtime = E2BRuntime(E2BConfig())
    runtime._sandbox = FakeSandbox()
    with pytest.raises(SandboxError, match="over the 3 byte limit"):
        await runtime.read("file", max_bytes=3)
