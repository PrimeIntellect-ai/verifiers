from types import SimpleNamespace
from typing import ClassVar

import pytest

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeProcess, PrimeRuntime


class _FakeTransport:
    instances: ClassVar[list["_FakeTransport"]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    async def aclose(self) -> None:
        self.closed = True


class _FakeHTTPClient:
    def __init__(self, transport=None) -> None:
        self.transport = transport


class _FakeConnectClient:
    unary_transports: ClassVar[list[object]] = []

    def __init__(self, base_url: str, *, http_client=None) -> None:
        self.base_url = base_url
        self.http_client = http_client

    def execute_server_stream(self, *, request, method, headers, timeout_ms):
        del request, method, headers, timeout_ms
        return SimpleNamespace(kind="stream")

    async def execute_unary(self, *, request, method, headers, timeout_ms):
        del request, method, headers, timeout_ms
        self.unary_transports.append(self.http_client.transport)

    async def close(self) -> None:
        pass


async def _empty_stream():
    return
    yield b""


class _FakeSDKProcess:
    fail_create = False

    def __init__(self, write_stdin) -> None:
        self.stdout = _empty_stream()
        self.stderr = _empty_stream()
        self._write_stdin = write_stdin
        self.closed = False

    @classmethod
    async def _create(cls, stream_client, stream, write_stdin, send_signal):
        del stream, send_signal
        assert stream_client.http_client is not None
        if cls.fail_create:
            raise RuntimeError("process stream refused")
        return cls(write_stdin)

    async def write_stdin(self, data: bytes) -> None:
        await self._write_stdin(4242, data)

    async def aclose(self) -> None:
        self.closed = True


def _runtime(monkeypatch) -> PrimeRuntime:
    import connectrpc.client
    import prime_sandboxes.process
    import pyqwest

    monkeypatch.setattr(pyqwest, "HTTPTransport", _FakeTransport)
    monkeypatch.setattr(pyqwest, "Client", _FakeHTTPClient)
    monkeypatch.setattr(connectrpc.client, "ConnectClient", _FakeConnectClient)
    monkeypatch.setattr(prime_sandboxes.process, "AsyncSandboxProcess", _FakeSDKProcess)
    _FakeConnectClient.unary_transports = []
    _FakeSDKProcess.fail_create = False
    _FakeTransport.instances = []

    class AuthCache:
        async def get_or_refresh(self, sandbox_id: str) -> dict:
            assert sandbox_id == "sandbox"
            return {
                "gateway_url": "https://gateway.test/",
                "user_ns": "ns",
                "job_id": "job",
                "token": "token",
            }

    class Client:
        _auth_cache = AuthCache()

        async def _should_retry_401(self, sandbox_id: str, reauthed: bool) -> bool:
            del sandbox_id, reauthed
            return False

    runtime = PrimeRuntime(PrimeConfig(vm=True), name="test")
    runtime._client = Client()
    runtime.info.id = "sandbox"
    return runtime


@pytest.mark.asyncio
async def test_prime_live_processes_get_one_transport_each(monkeypatch) -> None:
    runtime = _runtime(monkeypatch)

    first = await runtime.open_process(["cat"], {})
    second = await runtime.open_process(["cat"], {})

    assert isinstance(first, PrimeProcess) and isinstance(second, PrimeProcess)
    assert first._transport is not second._transport
    assert first._transport.transport.kwargs == {"tls_include_system_certs": True}

    await first.write(b"ping")
    await second.write(b"pong")
    assert _FakeConnectClient.unary_transports == [
        first._transport.transport,
        second._transport.transport,
    ]

    await first.aclose()
    assert first._process.closed
    assert first._transport.transport.closed
    assert not second._transport.transport.closed


@pytest.mark.asyncio
async def test_prime_open_process_closes_transport_on_failure(monkeypatch) -> None:
    runtime = _runtime(monkeypatch)
    _FakeSDKProcess.fail_create = True

    with pytest.raises(SandboxError, match="process stream refused"):
        await runtime.open_process(["cat"], {})

    assert [transport.closed for transport in _FakeTransport.instances] == [True]


@pytest.mark.asyncio
async def test_prime_process_aclose_releases_transport_when_close_fails() -> None:
    class SDKProcess:
        stdout = _empty_stream()
        stderr = _empty_stream()

        async def aclose(self) -> None:
            raise RuntimeError("stream already dead")

    transport = _FakeTransport()
    process = PrimeProcess(SDKProcess(), transport)

    with pytest.raises(RuntimeError, match="stream already dead"):
        await process.aclose()

    assert transport.closed


@pytest.mark.asyncio
async def test_acp_stop_always_closes_runtime_process() -> None:
    from verifiers.v1.acp import ACPHarnessSession
    from verifiers.v1.task import TaskData
    from verifiers.v1.trace import Trace

    class Process:
        stdout = _empty_stream()
        stderr = _empty_stream()

        def __init__(self) -> None:
            self.closed = False

        async def write(self, data: bytes) -> None:
            del data

        async def wait(self) -> int:
            return 0

        async def terminate(self) -> None:
            pass

        async def kill(self) -> None:
            pass

        async def aclose(self) -> None:
            self.closed = True

    session = ACPHarnessSession(
        SimpleNamespace(config=SimpleNamespace(id="rlm")),
        SimpleNamespace(),
        Trace.model_construct(id="trace"),
        SimpleNamespace(),
        "http://intercept",
        "secret",
        {},
        TaskData(prompt="hello"),
        env={},
        command=["rlm"],
        prompt="hello",
        system_prompt=None,
        session_meta=None,
    )
    process = Process()
    session._process = process

    await session._stop(graceful=False)

    assert process.closed
    assert session._process is None
