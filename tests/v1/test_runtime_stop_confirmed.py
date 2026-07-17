import asyncio
from types import SimpleNamespace

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
    else:
        asyncio.run(runtime.stop_confirmed())
        assert runtime._client is None
        assert client.closed is True
    assert client.deleted == ["prime-runtime-id"]


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
    else:
        asyncio.run(runtime.stop_confirmed())
        assert runtime._sandbox is None
    assert terminate.calls == 1


def test_confirmed_stop_fails_when_remote_handle_was_consumed_without_confirmation():
    prime = PrimeRuntime(PrimeConfig())
    prime.info.id = "prime-runtime-id"
    with pytest.raises(RuntimeError, match="live client"):
        asyncio.run(prime.stop_confirmed())

    modal = ModalRuntime(ModalConfig())
    modal.info.id = "modal-runtime-id"
    with pytest.raises(RuntimeError, match="live handle"):
        asyncio.run(modal.stop_confirmed())


class AsyncCallable:
    def __init__(self, function):
        self.function = function
        self.aio = self

    async def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def test_prime_start_forwards_egress_idle_timeout_and_hard_lifetime(monkeypatch):
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
            network_access=False,
            idle_timeout=61,
            timeout=121,
            labels=["emulatorbench"],
        )
    )
    asyncio.run(runtime.start())
    assert runtime.info.id == "prime-provider-id"
    assert len(requests) == 1
    request = requests[0]
    assert request["network_access"] is False
    assert request["idle_timeout_minutes"] == 2
    assert request["timeout_minutes"] == 3
    assert request["labels"] == ["emulatorbench"]


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
    asyncio.run(runtime.start())
    assert runtime.info.id == "modal-provider-id"
    create = next(row for row in calls if row[0] == "create")
    kwargs = create[2]
    assert kwargs["block_network"] is True
    assert kwargs["timeout"] == 121
    assert kwargs["encrypted_ports"]
