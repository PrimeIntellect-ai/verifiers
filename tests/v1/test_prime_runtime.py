from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import ValidationError

from verifiers.v1.runtimes.prime import PrimeConfig, PrimeProcess, PrimeRuntime


def test_prime_lifetime_timeout_is_configurable() -> None:
    config = PrimeConfig(vm=True, lifetime_timeout=600)

    assert config.lifetime_timeout == 600
    assert config.idle_timeout == 3_600


def test_prime_idle_timeout_cannot_exceed_lifetime() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        PrimeConfig(lifetime_timeout=3_600, idle_timeout=3_601)


@pytest.mark.asyncio
async def test_prime_runtime_uses_configured_lifetime(monkeypatch) -> None:
    import prime_sandboxes

    class Client:
        def __init__(self) -> None:
            self.request = None
            self.run_timeout = None

        async def create(self, request):
            self.request = request
            return SimpleNamespace(id="sandbox", pending_image_build_id=None)

        async def wait_for_creation(self, sandbox_id: str) -> None:
            assert sandbox_id == "sandbox"

        async def execute_command(self, sandbox_id: str, command: str):
            del sandbox_id, command
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        async def run_background_job(self, sandbox_id: str, command: str, **kwargs):
            del sandbox_id, command
            self.run_timeout = kwargs["timeout"]
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

    client = Client()
    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", lambda: client)
    runtime = PrimeRuntime(PrimeConfig(vm=True, lifetime_timeout=43_200), name="test")

    await runtime.start()
    await runtime.run(["true"], {})

    assert client.request.timeout_minutes == 720
    assert client.request.idle_timeout_minutes is None
    assert client.run_timeout == 43_200


class _FakeTransport:
    instances: ClassVar[list["_FakeTransport"]] = []

    def __init__(self) -> None:
        self.closed = False
        _FakeTransport.instances.append(self)

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
        _FakeConnectClient.unary_transports.append(self.http_client.transport)

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


def _transport_isolated_runtime(monkeypatch) -> PrimeRuntime:
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
    # The gateway caps one HTTP/2 connection at 100 concurrent streams and the
    # SDK funnels every command-session RPC over the shared default transport.
    # Each live process must own its transport, and its stdin RPCs must ride
    # that same transport, so processes never queue behind each other.
    runtime = _transport_isolated_runtime(monkeypatch)

    first = await runtime.open_process(["cat"], {})
    second = await runtime.open_process(["cat"], {})

    assert isinstance(first, PrimeProcess) and isinstance(second, PrimeProcess)
    assert first._transport is not second._transport

    await first.write(b"ping")
    await second.write(b"pong")
    assert _FakeConnectClient.unary_transports == [
        first._transport,
        second._transport,
    ]

    await first.aclose()
    assert first._process.closed
    assert first._transport.closed
    assert not second._transport.closed


@pytest.mark.asyncio
async def test_prime_open_process_closes_its_transport_on_failure(
    monkeypatch,
) -> None:
    from verifiers.v1.errors import SandboxError

    runtime = _transport_isolated_runtime(monkeypatch)
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
