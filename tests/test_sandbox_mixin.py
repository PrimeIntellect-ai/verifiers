import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx
from prime_sandboxes import (
    APIError,
    CommandTimeoutError,
    DownloadTimeoutError,
    SandboxOOMError,
    SandboxTimeoutError,
    UploadTimeoutError,
)

import verifiers as vf
from verifiers.envs.experimental.sandbox_mixin import (
    SandboxCreationError,
    SandboxMixin,
    SandboxMonitorRubric,
    SandboxNotReadyError,
    SandboxSetupError,
    SandboxTimeouts,
    SandboxVMUnsupportedError,
    ThreadedAsyncSandboxClient,
    is_retryable_sandbox_api_error,
    is_retryable_sandbox_read_error,
)

MODULE = "verifiers.envs.experimental.sandbox_mixin"


class ConcreteMixin(SandboxMixin):
    """Minimal concrete class for testing the mixin."""

    def __init__(self, **kwargs):
        self.init_sandbox_client(**kwargs)


@pytest.fixture
def mixin():
    obj = ConcreteMixin(max_retries=1, base_delay=0.01)
    obj.logger = MagicMock()
    obj.active_sandboxes = {"sb-existing"}
    return obj


# ── init_sandbox_client ──────────────────────────────────────────────


def test_init_creates_client_and_retry():
    obj = ConcreteMixin(max_retries=1, base_delay=0.01)
    assert isinstance(obj.active_sandboxes, set) and len(obj.active_sandboxes) == 0
    assert isinstance(obj.sandbox_client, ThreadedAsyncSandboxClient)
    assert callable(obj.with_retry)


@pytest.mark.parametrize(
    "exception",
    [
        UploadTimeoutError("sb", "/tmp/file", 300),
        DownloadTimeoutError("sb", "/tmp/file", 300),
        CommandTimeoutError("sb", "echo hi", 30),
        httpx.ReadTimeout("timed out"),
        APIError("Upload failed: HTTP 503: retry me"),
        APIError("Upload failed: ConnectError at POST /upload: boom"),
    ],
)
def test_retryable_sandbox_read_error_matches_current_sdk_exceptions(exception):
    assert is_retryable_sandbox_read_error(exception) is True


def test_retryable_sandbox_api_error_ignores_non_retryable_api_error():
    assert (
        is_retryable_sandbox_api_error(APIError("Upload failed: HTTP 400: nope"))
        is False
    )


# ── create_sandbox ───────────────────────────────────────────────────


def _container_request(**overrides):
    """Build a container (non-VM) request MagicMock with sensible defaults."""
    defaults = {"vm": False, "gpu_count": 0, "gpu_type": None}
    defaults.update(overrides)
    return MagicMock(**defaults)


def test_create_sandbox_success(mixin):
    sandbox_obj = MagicMock(id="sb-1")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock()

    state = {}
    result = asyncio.run(mixin.create_sandbox(state, request=_container_request()))

    assert result == "sb-1"
    assert state["sandbox_id"] == "sb-1"
    assert "sb-1" in mixin.active_sandboxes
    mixin.sandbox_client.wait_for_creation.assert_called_once_with(
        "sb-1",
        max_attempts=120,
    )


def test_create_sandbox_creation_fails(mixin):
    mixin.sandbox_client.create = AsyncMock(side_effect=Exception("boom"))

    with pytest.raises(SandboxCreationError):
        asyncio.run(mixin.create_sandbox({}, request=_container_request()))


def test_create_sandbox_max_retries_is_true_retry_count():
    obj = ConcreteMixin(max_retries=1, base_delay=0.01)
    obj.logger = MagicMock()
    sandbox_obj = MagicMock(id="sb-retry")
    obj.sandbox_client.create = AsyncMock(side_effect=[Exception("boom"), sandbox_obj])
    obj.sandbox_client.wait_for_creation = AsyncMock()

    result = asyncio.run(obj.create_sandbox({}, request=_container_request()))

    assert result == "sb-retry"
    assert obj.sandbox_client.create.await_count == 2


def test_create_sandbox_not_ready(mixin):
    sandbox_obj = MagicMock(id="sb-2")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock(
        side_effect=Exception("not ready")
    )

    with pytest.raises(SandboxNotReadyError):
        asyncio.run(mixin.create_sandbox({}, request=_container_request()))

    # Sandbox was added before wait_for_creation, so it should still be tracked.
    assert "sb-2" in mixin.active_sandboxes


def test_create_sandbox_wait_for_creation_respects_custom_attempts():
    obj = ConcreteMixin(
        max_retries=1,
        base_delay=0.01,
        sandbox_wait_for_creation_max_attempts=7,
    )
    obj.logger = MagicMock()
    sandbox_obj = MagicMock(id="sb-custom")
    obj.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    obj.sandbox_client.wait_for_creation = AsyncMock()

    asyncio.run(obj.create_sandbox({}, request=_container_request()))

    obj.sandbox_client.wait_for_creation.assert_called_once_with(
        "sb-custom",
        max_attempts=7,
    )


def test_create_sandbox_setup_fails(mixin):
    sandbox_obj = MagicMock(id="sb-3")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock()

    async def bad_setup(state):
        raise RuntimeError("setup boom")

    mixin.post_sandbox_setup = bad_setup

    with pytest.raises(SandboxSetupError):
        asyncio.run(mixin.create_sandbox({}, request=_container_request()))


def test_create_sandbox_setup_sandbox_error_passthrough(mixin):
    sandbox_obj = MagicMock(id="sb-4")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock()

    async def sandbox_err_setup(state):
        raise vf.SandboxError("custom sandbox error")

    mixin.post_sandbox_setup = sandbox_err_setup

    with pytest.raises(vf.SandboxError, match="custom sandbox error"):
        asyncio.run(mixin.create_sandbox({}, request=_container_request()))


# ── post_sandbox_setup ───────────────────────────────────────────────


def test_post_sandbox_setup_is_noop(mixin):
    result = asyncio.run(mixin.post_sandbox_setup({}))
    assert result is None


# ── delete_sandbox ───────────────────────────────────────────────────


def test_delete_sandbox_success(mixin):
    mixin.sandbox_client.delete = AsyncMock()

    asyncio.run(mixin.delete_sandbox("sb-existing"))

    assert "sb-existing" not in mixin.active_sandboxes


def test_delete_sandbox_failure_logs_warning(mixin):
    mixin.sandbox_client.delete = AsyncMock(side_effect=Exception("fail"))

    asyncio.run(mixin.delete_sandbox("sb-existing"))

    mixin.logger.warning.assert_called_once()
    # No exception raised — it was swallowed.


# ── bulk_delete_sandboxes ────────────────────────────────────────────


def test_bulk_delete_success(mixin):
    mixin.active_sandboxes = {"sb-a", "sb-b", "sb-c"}
    mixin.sandbox_client.bulk_delete = AsyncMock()

    asyncio.run(mixin.bulk_delete_sandboxes(["sb-a", "sb-c"]))

    mixin.sandbox_client.bulk_delete.assert_called_once_with(["sb-a", "sb-c"])
    assert mixin.active_sandboxes == {"sb-b"}


def test_bulk_delete_failure(mixin):
    mixin.active_sandboxes = {"sb-a", "sb-b"}
    mixin.sandbox_client.bulk_delete = AsyncMock(side_effect=Exception("bulk fail"))

    asyncio.run(mixin.bulk_delete_sandboxes(["sb-a"]))

    mixin.logger.error.assert_called_once()
    assert mixin.active_sandboxes == {"sb-a", "sb-b"}


# ── run_background_job ───────────────────────────────────────────────


def test_run_background_job_success(mixin):
    results = MagicMock(completed=True)
    mixin.sandbox_client.run_background_job = AsyncMock(return_value=results)

    state = {"sandbox_id": "sb-1"}
    ret = asyncio.run(mixin.run_background_job(state, command="echo hi", timeout=10))
    assert ret is results


def test_run_background_job_command_timeout(mixin):
    mixin.sandbox_client.run_background_job = AsyncMock(
        side_effect=CommandTimeoutError(sandbox_id="sb-1", command="cmd", timeout=5)
    )

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(CommandTimeoutError):
        asyncio.run(mixin.run_background_job(state, command="cmd", timeout=10))


def test_run_background_job_oom(mixin):
    mixin.sandbox_client.run_background_job = AsyncMock(
        side_effect=SandboxOOMError(sandbox_id="sb-1")
    )

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(vf.SandboxError):
        asyncio.run(mixin.run_background_job(state, command="cmd", timeout=10))
    assert state["sandbox_oom"] is True


def test_run_background_job_sandbox_timeout(mixin):
    mixin.sandbox_client.run_background_job = AsyncMock(
        side_effect=SandboxTimeoutError(sandbox_id="sb-1")
    )

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(vf.SandboxError):
        asyncio.run(mixin.run_background_job(state, command="cmd", timeout=10))
    assert state["sandbox_timeout"] is True


# ── teardown_sandboxes ──────────────────────────────────────────────


def test_teardown_sandboxes_empty(mixin):
    mixin.active_sandboxes = set()

    with patch(f"{MODULE}.SandboxClient") as mock_sync:
        mixin.teardown_sandboxes()
        mock_sync.assert_not_called()


def test_teardown_sandboxes_batched(mixin):
    mixin.active_sandboxes = {f"sb-{i}" for i in range(150)}

    mock_sync_client = MagicMock()
    with (
        patch(f"{MODULE}.SandboxClient", return_value=mock_sync_client),
        patch(f"{MODULE}.APIClient"),
    ):
        mixin.teardown_sandboxes()

    assert mock_sync_client.bulk_delete.call_count == 2
    # All IDs should have been cleared.
    assert len(mixin.active_sandboxes) == 0


def test_teardown_sandboxes_partial_failure(mixin):
    mixin.active_sandboxes = {f"sb-{i}" for i in range(150)}

    mock_sync_client = MagicMock()
    # First batch fails, second succeeds.
    mock_sync_client.bulk_delete.side_effect = [
        Exception("batch fail"),
        None,
    ]

    with (
        patch(f"{MODULE}.SandboxClient", return_value=mock_sync_client),
        patch(f"{MODULE}.APIClient"),
    ):
        mixin.teardown_sandboxes()

    mixin.logger.warning.assert_called_once()
    # The second batch should have been cleared; the first batch remains.
    assert len(mixin.active_sandboxes) == 100


# ── teardown_sandbox_client ──────────────────────────────────────────


def test_teardown_sandbox_client(mixin):
    mixin.sandbox_client = MagicMock()

    mixin.teardown_sandbox_client()

    mixin.sandbox_client.teardown.assert_called_once()


def test_teardown_mixin_sandboxes_handler_calls_helper(mixin):
    mixin.teardown_sandboxes = MagicMock()

    asyncio.run(mixin.teardown_mixin_sandboxes())

    mixin.teardown_sandboxes.assert_called_once()


def test_teardown_mixin_sandbox_client_handler_calls_helper(mixin):
    mixin.teardown_sandbox_client = MagicMock()

    asyncio.run(mixin.teardown_mixin_sandbox_client())

    mixin.teardown_sandbox_client.assert_called_once()


def test_mixin_teardown_handlers_are_decorated():
    assert getattr(ConcreteMixin.teardown_mixin_sandboxes, "teardown", False) is True
    assert getattr(ConcreteMixin.teardown_mixin_sandboxes, "teardown_priority") == -10

    assert (
        getattr(ConcreteMixin.teardown_mixin_sandbox_client, "teardown", False) is True
    )
    assert (
        getattr(ConcreteMixin.teardown_mixin_sandbox_client, "teardown_priority") == -20
    )


# ── SandboxTimeouts ──────────────────────────────────────────────────


def test_sandbox_timeouts_defaults(mixin):
    assert isinstance(mixin.timeouts, SandboxTimeouts)
    assert mixin.timeouts.read_file == 10
    assert mixin.timeouts.extract == 60
    assert mixin.timeouts.poll == 60
    assert mixin.timeouts.mkdir == 10
    # Sandbox sidecar deserializes `timeout` as u64, so fields must be int.
    for field_name in ("read_file", "extract", "poll", "mkdir"):
        assert isinstance(getattr(mixin.timeouts, field_name), int), field_name


def test_sandbox_timeouts_override():
    custom = SandboxTimeouts(read_file=5, extract=120)
    obj = ConcreteMixin(max_retries=1, base_delay=0.01, timeouts=custom)
    assert obj.timeouts.read_file == 5
    assert obj.timeouts.extract == 120
    # Untouched fields remain at defaults.
    assert obj.timeouts.poll == 60
    assert obj.timeouts.mkdir == 10


def test_sandbox_timeouts_is_frozen():
    t = SandboxTimeouts()
    with pytest.raises(Exception):
        t.read_file = 99  # type: ignore[misc]


def test_read_file_uses_timeouts_default(mixin):
    fake_result = MagicMock(content="hello")
    mixin.sandbox_client.read_file = AsyncMock(return_value=fake_result)

    out = asyncio.run(mixin.read_file("sb-1", "/tmp/x"))

    assert out == "hello"
    mixin.sandbox_client.read_file.assert_awaited_once_with(
        "sb-1", "/tmp/x", timeout=mixin.timeouts.read_file
    )


def test_read_file_uses_timeouts_default_when_configured():
    obj = ConcreteMixin(
        max_retries=1,
        base_delay=0.01,
        timeouts=SandboxTimeouts(read_file=7),
    )
    obj.logger = MagicMock()
    fake_result = MagicMock(content="hi")
    obj.sandbox_client.read_file = AsyncMock(return_value=fake_result)

    asyncio.run(obj.read_file("sb-1", "/tmp/y"))

    obj.sandbox_client.read_file.assert_awaited_once_with("sb-1", "/tmp/y", timeout=7)


def test_read_file_explicit_override_beats_default(mixin):
    fake_result = MagicMock(content="hello")
    mixin.sandbox_client.read_file = AsyncMock(return_value=fake_result)

    asyncio.run(mixin.read_file("sb-1", "/tmp/x", timeout=123))

    mixin.sandbox_client.read_file.assert_awaited_once_with(
        "sb-1", "/tmp/x", timeout=123
    )


def test_upload_bundle_uses_extract_timeout(mixin):
    # Avoid exercising real upload/tarball code paths.
    async def _noop_upload(sandbox_id, remote_path, local_path):
        return None

    mixin.upload_file = _noop_upload
    mixin.sandbox_client.execute_command = AsyncMock(
        return_value=MagicMock(exit_code=0, stderr="")
    )

    asyncio.run(mixin.upload_bundle("sb-1", {"a.txt": "x"}, "/tmp/dest"))

    _, kwargs = mixin.sandbox_client.execute_command.call_args
    assert kwargs["timeout"] == mixin.timeouts.extract


def test_upload_bundle_respects_custom_extract_timeout():
    obj = ConcreteMixin(
        max_retries=1,
        base_delay=0.01,
        timeouts=SandboxTimeouts(extract=180),
    )
    obj.logger = MagicMock()

    async def _noop_upload(sandbox_id, remote_path, local_path):
        return None

    obj.upload_file = _noop_upload  # type: ignore[method-assign]
    obj.sandbox_client.execute_command = AsyncMock(
        return_value=MagicMock(exit_code=0, stderr="")
    )

    asyncio.run(obj.upload_bundle("sb-1", {"a.txt": "x"}, "/tmp/dest"))

    _, kwargs = obj.sandbox_client.execute_command.call_args
    assert kwargs["timeout"] == 180


def test_cli_agent_env_poll_uses_timeouts_poll():
    from verifiers.envs.experimental.cli_agent_env import CliAgentEnv

    # Bypass __init__ to avoid spinning up interception server / MultiTurnEnv
    # machinery; exercise only the timeouts plumbing.
    env = CliAgentEnv.__new__(CliAgentEnv)
    SandboxMixin.init_sandbox_client(
        env,
        max_retries=1,
        base_delay=0.01,
        timeouts=SandboxTimeouts(poll=90),
    )

    assert env.timeouts.poll == 90
    assert env.timeouts.read_file == 10  # untouched


def test_cli_agent_env_defaults_match_hardcodes():
    from verifiers.envs.experimental.cli_agent_env import CliAgentEnv

    env = CliAgentEnv.__new__(CliAgentEnv)
    SandboxMixin.init_sandbox_client(env, max_retries=1, base_delay=0.01)

    # Defaults match the pre-refactor literal values.
    assert env.timeouts.read_file == 10.0
    assert env.timeouts.extract == 60.0
    assert env.timeouts.poll == 60.0
    assert env.timeouts.mkdir == 10.0


# ── build_sandbox_request ────────────────────────────────────────────


def test_build_sandbox_request_container_defaults(mixin):
    req = mixin.build_sandbox_request(name="rollout-1", docker_image="python:3.12")
    assert req.name == "rollout-1"
    assert req.docker_image == "python:3.12"
    assert req.vm is False
    assert req.gpu_count == 0
    assert req.gpu_type is None
    assert req.labels == []


def test_build_sandbox_request_vm_with_gpu(mixin):
    req = mixin.build_sandbox_request(
        name="rollout-vm",
        docker_image="my-vm-image:latest",
        vm=True,
        gpu_count=1,
        gpu_type="H100_80GB",
    )
    assert req.vm is True
    assert req.gpu_count == 1
    assert req.gpu_type == "H100_80GB"


def test_build_sandbox_request_gpu_without_vm_raises(mixin):
    with pytest.raises(vf.SandboxError, match="Invalid CreateSandboxRequest"):
        mixin.build_sandbox_request(
            name="bad",
            docker_image="img",
            vm=False,
            gpu_count=1,
            gpu_type="H100_80GB",
        )


def test_build_sandbox_request_gpu_count_without_type_raises(mixin):
    with pytest.raises(vf.SandboxError, match="Invalid CreateSandboxRequest"):
        mixin.build_sandbox_request(
            name="bad",
            docker_image="img",
            vm=True,
            gpu_count=1,
        )


def test_build_sandbox_request_gpu_type_without_count_raises(mixin):
    with pytest.raises(vf.SandboxError, match="Invalid CreateSandboxRequest"):
        mixin.build_sandbox_request(
            name="bad",
            docker_image="img",
            vm=True,
            gpu_count=0,
            gpu_type="H100_80GB",
        )


# ── VM-aware create_sandbox ──────────────────────────────────────────


def test_create_sandbox_populates_vm_state_fields(mixin):
    sandbox_obj = MagicMock(id="sb-vm")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock()

    request = MagicMock(vm=True, gpu_count=2, gpu_type="H100_80GB")
    state: dict = {}
    asyncio.run(mixin.create_sandbox(state, request=request))

    assert state["sandbox_id"] == "sb-vm"
    assert state["sandbox_is_vm"] is True
    assert state["sandbox_gpu_count"] == 2
    assert state["sandbox_gpu_type"] == "H100_80GB"


def test_create_sandbox_container_sets_vm_false(mixin):
    sandbox_obj = MagicMock(id="sb-cont")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock()

    request = MagicMock(vm=False, gpu_count=0, gpu_type=None)
    state: dict = {}
    asyncio.run(mixin.create_sandbox(state, request=request))

    assert state["sandbox_is_vm"] is False
    assert state["sandbox_gpu_count"] == 0
    assert state["sandbox_gpu_type"] is None


def test_create_sandbox_vm_uses_vm_wait_attempts():
    obj = ConcreteMixin(
        max_retries=1,
        base_delay=0.01,
        sandbox_wait_for_creation_max_attempts=7,
        vm_sandbox_wait_for_creation_max_attempts=42,
    )
    obj.logger = MagicMock()
    sandbox_obj = MagicMock(id="sb-vm-wait")
    obj.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    obj.sandbox_client.wait_for_creation = AsyncMock()

    request = MagicMock(vm=True, gpu_count=0, gpu_type=None)
    asyncio.run(obj.create_sandbox({}, request=request))

    obj.sandbox_client.wait_for_creation.assert_called_once_with(
        "sb-vm-wait", max_attempts=42
    )


def test_create_sandbox_container_uses_container_wait_attempts():
    obj = ConcreteMixin(
        max_retries=1,
        base_delay=0.01,
        sandbox_wait_for_creation_max_attempts=7,
        vm_sandbox_wait_for_creation_max_attempts=42,
    )
    obj.logger = MagicMock()
    sandbox_obj = MagicMock(id="sb-cont-wait")
    obj.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    obj.sandbox_client.wait_for_creation = AsyncMock()

    request = MagicMock(vm=False, gpu_count=0, gpu_type=None)
    asyncio.run(obj.create_sandbox({}, request=request))

    obj.sandbox_client.wait_for_creation.assert_called_once_with(
        "sb-cont-wait", max_attempts=7
    )


def test_create_sandbox_vm_uses_vm_rate_limiter():
    obj = ConcreteMixin(max_retries=1, base_delay=0.01)
    obj.logger = MagicMock()
    # Replace both limiters with mocks that record acquire() calls.
    cont_limiter = MagicMock()
    cont_limiter.acquire = AsyncMock()
    vm_limiter = MagicMock()
    vm_limiter.acquire = AsyncMock()
    obj.sandbox_creation_rate_limiter = cont_limiter
    obj.vm_sandbox_creation_rate_limiter = vm_limiter

    sandbox_obj = MagicMock(id="sb-vm-rl")
    obj.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    obj.sandbox_client.wait_for_creation = AsyncMock()

    request = MagicMock(vm=True, gpu_count=0, gpu_type=None)
    asyncio.run(obj.create_sandbox({}, request=request))

    vm_limiter.acquire.assert_awaited_once()
    cont_limiter.acquire.assert_not_awaited()


def test_create_sandbox_container_uses_container_rate_limiter():
    obj = ConcreteMixin(max_retries=1, base_delay=0.01)
    obj.logger = MagicMock()
    cont_limiter = MagicMock()
    cont_limiter.acquire = AsyncMock()
    vm_limiter = MagicMock()
    vm_limiter.acquire = AsyncMock()
    obj.sandbox_creation_rate_limiter = cont_limiter
    obj.vm_sandbox_creation_rate_limiter = vm_limiter

    sandbox_obj = MagicMock(id="sb-cont-rl")
    obj.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    obj.sandbox_client.wait_for_creation = AsyncMock()

    request = MagicMock(vm=False, gpu_count=0, gpu_type=None)
    asyncio.run(obj.create_sandbox({}, request=request))

    cont_limiter.acquire.assert_awaited_once()
    vm_limiter.acquire.assert_not_awaited()


def test_init_sandbox_client_disables_vm_rate_limiter_when_none():
    obj = ConcreteMixin(
        max_retries=1,
        base_delay=0.01,
        vm_sandbox_creations_per_minute=None,
    )
    assert obj.vm_sandbox_creation_rate_limiter is None
    # Container limiter still enabled by default.
    assert obj.sandbox_creation_rate_limiter is not None


# ── VM unsupported-op guards ────────────────────────────────────────


def test_expose_port_raises_on_vm(mixin):
    mixin.sandbox_client.expose = AsyncMock()
    state = {"sandbox_is_vm": True}

    with pytest.raises(SandboxVMUnsupportedError, match="Port exposure"):
        asyncio.run(mixin.expose_port(state, "sb-vm", 8080))

    mixin.sandbox_client.expose.assert_not_called()


def test_expose_port_delegates_on_container(mixin):
    mixin.sandbox_client.expose = AsyncMock(return_value="exposure")
    state = {"sandbox_is_vm": False}

    result = asyncio.run(
        mixin.expose_port(state, "sb-cont", 8080, name="web", protocol="HTTP")
    )
    assert result == "exposure"
    mixin.sandbox_client.expose.assert_awaited_once_with(
        "sb-cont", 8080, name="web", protocol="HTTP"
    )


def test_unexpose_port_raises_on_vm(mixin):
    mixin.sandbox_client.unexpose = AsyncMock()
    state = {"sandbox_is_vm": True}

    with pytest.raises(SandboxVMUnsupportedError, match="Port unexpose"):
        asyncio.run(mixin.unexpose_port(state, "sb-vm", "exp-1"))

    mixin.sandbox_client.unexpose.assert_not_called()


def test_unexpose_port_delegates_on_container(mixin):
    mixin.sandbox_client.unexpose = AsyncMock()
    state = {"sandbox_is_vm": False}

    asyncio.run(mixin.unexpose_port(state, "sb-cont", "exp-1"))

    mixin.sandbox_client.unexpose.assert_awaited_once_with("sb-cont", "exp-1")


def test_list_exposed_ports_raises_on_vm(mixin):
    mixin.sandbox_client.list_exposed_ports = AsyncMock()
    state = {"sandbox_is_vm": True}

    with pytest.raises(SandboxVMUnsupportedError, match="Port listing"):
        asyncio.run(mixin.list_exposed_ports(state, "sb-vm"))


def test_list_exposed_ports_delegates_on_container(mixin):
    mixin.sandbox_client.list_exposed_ports = AsyncMock(return_value="ports")
    state = {"sandbox_is_vm": False}

    result = asyncio.run(mixin.list_exposed_ports(state, "sb-cont"))
    assert result == "ports"
    mixin.sandbox_client.list_exposed_ports.assert_awaited_once_with("sb-cont")


def test_create_ssh_session_raises_on_vm(mixin):
    mixin.sandbox_client.create_ssh_session = AsyncMock()
    state = {"sandbox_is_vm": True}

    with pytest.raises(SandboxVMUnsupportedError, match="SSH"):
        asyncio.run(mixin.create_ssh_session(state, "sb-vm"))


def test_create_ssh_session_delegates_on_container(mixin):
    mixin.sandbox_client.create_ssh_session = AsyncMock(return_value="session")
    state = {"sandbox_is_vm": False}

    result = asyncio.run(mixin.create_ssh_session(state, "sb-cont", ttl_seconds=600))
    assert result == "session"
    mixin.sandbox_client.create_ssh_session.assert_awaited_once_with(
        "sb-cont", ttl_seconds=600
    )


# ── SandboxMonitorRubric VM metrics ──────────────────────────────────


def test_sandbox_monitor_rubric_reports_vm_metrics():
    rubric = SandboxMonitorRubric()
    state = {
        "sandbox_is_vm": True,
        "sandbox_gpu_count": 4,
    }
    assert asyncio.run(rubric.sandbox_is_vm(state)) == 1.0
    assert asyncio.run(rubric.sandbox_gpu_count(state)) == 4.0


def test_sandbox_monitor_rubric_vm_metrics_default_zero():
    rubric = SandboxMonitorRubric()
    assert asyncio.run(rubric.sandbox_is_vm({})) == 0.0
    assert asyncio.run(rubric.sandbox_gpu_count({})) == 0.0
