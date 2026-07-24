import asyncio
from types import SimpleNamespace

from prime_sandboxes.core.client import APIError

import pytest

from verifiers.v1.runtimes.modal import ModalConfig, ModalRuntime
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime


class FakePrimeClient:
    def __init__(self, error=None):
        self.error = error
        self.deleted = []
        self.closed = False

    async def delete(self, runtime_id):
        self.deleted.append(runtime_id)
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True


class FakeTerminate:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0
        self.aio = self

    async def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error


@pytest.mark.parametrize("fails", [False, True])
def test_prime_stop_confirmed_preserves_cleanup_state_on_failure(fails):
    runtime = PrimeRuntime(PrimeConfig())
    runtime.info.id = "prime-runtime-id"
    client = FakePrimeClient(RuntimeError("delete failed") if fails else None)
    runtime._client = client
    if fails:
        with pytest.raises(RuntimeError, match="delete failed"):
            asyncio.run(runtime.stop_confirmed())
        assert runtime._client is client
        client.error = None

    asyncio.run(runtime.stop_confirmed())
    assert runtime._client is None
    assert client.closed is True
    assert runtime.info.id == "prime-runtime-id"
    expected_calls = 2 if fails else 1
    assert client.deleted == ["prime-runtime-id"] * expected_calls

    asyncio.run(runtime.stop_confirmed())
    assert client.deleted == ["prime-runtime-id"] * expected_calls


def test_prime_stop_confirmed_treats_already_gone_sandbox_as_success():
    """A 404 from delete means the sandbox was already removed; stop_confirmed
    should treat it as confirmed success, not fail forever on retries."""
    runtime = PrimeRuntime(PrimeConfig())
    runtime.info.id = "prime-runtime-id"
    client = FakePrimeClient(APIError("HTTP 404: sandbox not found"))
    runtime._client = client

    asyncio.run(runtime.stop_confirmed())
    assert runtime._client is None
    assert client.closed is True
    assert runtime._confirmed_stop_id == "prime-runtime-id"
    assert client.deleted == ["prime-runtime-id"]

    # Idempotent: a second call is a no-op (no new delete attempt).
    asyncio.run(runtime.stop_confirmed())
    assert client.deleted == ["prime-runtime-id"]


def test_prime_stop_confirmed_sets_stopped_flag():
    """stop_confirmed must set stopped=True so borrow checks reject the dead runtime."""
    runtime = PrimeRuntime(PrimeConfig())
    runtime.info.id = "prime-runtime-id"
    client = FakePrimeClient()
    runtime._client = client
    assert runtime.stopped is False
    asyncio.run(runtime.stop_confirmed())
    assert runtime.stopped is True


def test_modal_stop_confirmed_sets_stopped_flag():
    """stop_confirmed must set stopped=True so borrow checks reject the dead runtime."""
    runtime = ModalRuntime(ModalConfig())
    runtime.info.id = "modal-runtime-id"
    terminate = FakeTerminate()
    sandbox = SimpleNamespace(terminate=terminate)
    runtime._sandbox = sandbox
    assert runtime.stopped is False
    asyncio.run(runtime.stop_confirmed())
    assert runtime.stopped is True


def test_prime_stop_confirmed_closes_client_when_provider_id_missing():
    """When start failed before setting info.id, stop_confirmed must close the
    live client before raising, so the client does not leak."""
    runtime = PrimeRuntime(PrimeConfig())
    runtime.info.id = None
    client = FakePrimeClient()
    runtime._client = client

    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="provider ID"):
        asyncio.run(runtime.stop_confirmed())
    assert runtime._client is None
    assert client.closed is True


def test_prime_stop_confirmed_rejects_loose_404_in_non_api_errors():
    """A non-APIError whose message happens to contain '404' must NOT be
    treated as a confirmed deletion — only the provider's own APIError with
    an HTTP 404 prefix qualifies."""
    runtime = PrimeRuntime(PrimeConfig())
    runtime.info.id = "prime-runtime-id"
    # RuntimeError with 404 in the message — must propagate, not be swallowed.
    client = FakePrimeClient(RuntimeError("connect to port 404 failed"))
    runtime._client = client

    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="connect to port 404 failed"):
        asyncio.run(runtime.stop_confirmed())
    assert runtime._client is client
    assert runtime._confirmed_stop_id is None


def test_prime_stop_confirmed_rejects_non_404_api_errors():
    """An APIError with a non-404 status code must propagate, not be treated
    as confirmed deletion."""
    runtime = PrimeRuntime(PrimeConfig())
    runtime.info.id = "prime-runtime-id"
    client = FakePrimeClient(APIError("HTTP 500: internal server error"))
    runtime._client = client

    import pytest as _pytest

    with _pytest.raises(APIError, match="HTTP 500"):
        asyncio.run(runtime.stop_confirmed())
    assert runtime._client is client
    assert runtime._confirmed_stop_id is None


@pytest.mark.parametrize("fails", [False, True])
def test_modal_stop_confirmed_preserves_cleanup_state_on_failure(fails):
    runtime = ModalRuntime(ModalConfig())
    runtime.info.id = "modal-runtime-id"
    terminate = FakeTerminate(RuntimeError("terminate failed") if fails else None)
    sandbox = SimpleNamespace(terminate=terminate)
    runtime._sandbox = sandbox
    if fails:
        with pytest.raises(RuntimeError, match="terminate failed"):
            asyncio.run(runtime.stop_confirmed())
        assert runtime._sandbox is sandbox
        terminate.error = None

    asyncio.run(runtime.stop_confirmed())
    assert runtime._sandbox is None
    assert runtime.info.id == "modal-runtime-id"
    expected_calls = 2 if fails else 1
    assert terminate.calls == expected_calls

    asyncio.run(runtime.stop_confirmed())
    assert terminate.calls == expected_calls


def test_confirmed_stop_fails_when_remote_handle_was_consumed_without_confirmation():
    prime = PrimeRuntime(PrimeConfig())
    prime.info.id = "prime-runtime-id"
    with pytest.raises(RuntimeError, match="live client"):
        asyncio.run(prime.stop_confirmed())

    modal = ModalRuntime(ModalConfig())
    modal.info.id = "modal-runtime-id"
    with pytest.raises(RuntimeError, match="live handle"):
        asyncio.run(modal.stop_confirmed())


def test_prime_idle_timeout_cannot_exceed_hard_lifetime():
    with pytest.raises(ValueError, match="must not exceed the hard sandbox timeout"):
        PrimeConfig(idle_timeout=121, timeout=120)


def test_prime_idle_timeout_must_be_positive_or_disabled():
    with pytest.raises(ValueError, match="must be positive or None"):
        PrimeConfig(idle_timeout=0)
    assert PrimeConfig(idle_timeout=None, timeout=60).idle_timeout is None


class AsyncCallable:
    def __init__(self, function):
        self.function = function
        self.aio = self

    async def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def test_prime_start_forwards_idle_timeout_and_hard_lifetime(monkeypatch):
    requests = []

    class CreateSandboxRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            requests.append(kwargs)

    class Client:
        async def create(self, request):
            assert isinstance(request, CreateSandboxRequest)
            return SimpleNamespace(id="prime-provider-id", pending_image_build_id=None)

        async def wait_for_creation(self, runtime_id):
            assert runtime_id == "prime-provider-id"

        async def execute_command(self, runtime_id, command):
            assert runtime_id == "prime-provider-id"
            assert "mkdir -p" in command

    monkeypatch.setitem(
        __import__("sys").modules,
        "prime_sandboxes",
        SimpleNamespace(
            AsyncSandboxClient=Client, CreateSandboxRequest=CreateSandboxRequest
        ),
    )
    runtime = PrimeRuntime(
        PrimeConfig(
            image="pinned@example",
            idle_timeout=61,
            timeout=121,
            labels=["generic-runtime-test"],
        )
    )
    runtime._confirmed_stop_id = "prior-runtime-id"
    asyncio.run(runtime.start())
    assert runtime.info.id == "prime-provider-id"
    assert runtime._confirmed_stop_id is None
    assert len(requests) == 1
    request = requests[0]
    assert "network_access" not in request
    assert request["idle_timeout_minutes"] == 2
    assert request["timeout_minutes"] == 3
    assert request["labels"] == ["generic-runtime-test"]


def test_modal_start_forwards_egress_block_and_hard_lifetime(monkeypatch):
    calls = []

    class Image:
        @classmethod
        def from_registry(cls, image):
            calls.append(("image", image))
            return cls()

        def entrypoint(self, value):
            calls.append(("entrypoint", value))
            return self

    filesystem = SimpleNamespace(
        make_directory=AsyncCallable(lambda path: calls.append(("mkdir", path)))
    )
    sandbox = SimpleNamespace(object_id="modal-provider-id", filesystem=filesystem)

    class App:
        lookup = AsyncCallable(lambda *args, **kwargs: SimpleNamespace(name=args[0]))

    class Sandbox:
        create = AsyncCallable(
            lambda *args, **kwargs: (calls.append(("create", args, kwargs)), sandbox)[1]
        )

    monkeypatch.setitem(
        __import__("sys").modules,
        "modal",
        SimpleNamespace(App=App, Sandbox=Sandbox, Image=Image),
    )
    runtime = ModalRuntime(
        ModalConfig(
            image="pinned@example",
            network_access=False,
            timeout=121,
            creates_per_sec=None,
        )
    )
    runtime._confirmed_stop_id = "prior-runtime-id"
    asyncio.run(runtime.start())
    assert runtime.info.id == "modal-provider-id"
    assert runtime._confirmed_stop_id is None
    create = next(row for row in calls if row[0] == "create")
    kwargs = create[2]
    assert kwargs["block_network"] is True
    assert kwargs["timeout"] == 121
    assert kwargs["encrypted_ports"]
