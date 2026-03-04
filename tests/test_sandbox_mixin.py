import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prime_sandboxes import CommandTimeoutError, SandboxOOMError, SandboxTimeoutError

import verifiers as vf
from verifiers.envs.experimental.sandbox_mixin import (
    SandboxCreationError,
    SandboxMixin,
    SandboxNotReadyError,
    SandboxSetupError,
    ThreadedAsyncSandboxClient,
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


# ── create_sandbox ───────────────────────────────────────────────────


def test_create_sandbox_success(mixin):
    sandbox_obj = MagicMock(id="sb-1")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock()

    state = {}
    result = asyncio.run(mixin.create_sandbox(state, request=MagicMock()))

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
        asyncio.run(mixin.create_sandbox({}, request=MagicMock()))


def test_create_sandbox_not_ready(mixin):
    sandbox_obj = MagicMock(id="sb-2")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock(
        side_effect=Exception("not ready")
    )

    with pytest.raises(SandboxNotReadyError):
        asyncio.run(mixin.create_sandbox({}, request=MagicMock()))

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

    asyncio.run(obj.create_sandbox({}, request=MagicMock()))

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
        asyncio.run(mixin.create_sandbox({}, request=MagicMock()))


def test_create_sandbox_setup_sandbox_error_passthrough(mixin):
    sandbox_obj = MagicMock(id="sb-4")
    mixin.sandbox_client.create = AsyncMock(return_value=sandbox_obj)
    mixin.sandbox_client.wait_for_creation = AsyncMock()

    async def sandbox_err_setup(state):
        raise vf.SandboxError("custom sandbox error")

    mixin.post_sandbox_setup = sandbox_err_setup

    with pytest.raises(vf.SandboxError, match="custom sandbox error"):
        asyncio.run(mixin.create_sandbox({}, request=MagicMock()))


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
    job = MagicMock()
    results = MagicMock(completed=True)
    mixin.sandbox_client.start_background_job = AsyncMock(return_value=job)
    mixin.sandbox_client.get_background_job = AsyncMock(return_value=results)

    state = {"sandbox_id": "sb-1"}
    ret = asyncio.run(mixin.run_background_job(state, command="echo hi", timeout=10))
    assert ret is results


def test_run_background_job_command_timeout(mixin):
    mixin.sandbox_client.start_background_job = AsyncMock(
        side_effect=CommandTimeoutError(sandbox_id="sb-1", command="cmd", timeout=5)
    )

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(vf.SandboxError):
        asyncio.run(mixin.run_background_job(state, command="cmd", timeout=10))


def test_run_background_job_oom_on_start(mixin):
    mixin.sandbox_client.start_background_job = AsyncMock(
        side_effect=SandboxOOMError(sandbox_id="sb-1")
    )

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(vf.SandboxError):
        asyncio.run(mixin.run_background_job(state, command="cmd", timeout=10))
    assert state["sandbox_oom"] is True


def test_run_background_job_timeout_on_start(mixin):
    mixin.sandbox_client.start_background_job = AsyncMock(
        side_effect=SandboxTimeoutError(sandbox_id="sb-1")
    )

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(vf.SandboxError):
        asyncio.run(mixin.run_background_job(state, command="cmd", timeout=10))
    assert state["sandbox_timeout"] is True


def test_run_background_job_oom_during_poll(mixin):
    job = MagicMock()
    mixin.sandbox_client.start_background_job = AsyncMock(return_value=job)
    mixin.sandbox_client.get_background_job = AsyncMock(
        side_effect=SandboxOOMError(sandbox_id="sb-1")
    )

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(vf.SandboxError):
        asyncio.run(mixin.run_background_job(state, command="cmd", timeout=10))
    assert state["sandbox_oom"] is True


def test_run_background_job_timeout_during_poll(mixin):
    job = MagicMock()
    mixin.sandbox_client.start_background_job = AsyncMock(return_value=job)
    mixin.sandbox_client.get_background_job = AsyncMock(
        side_effect=SandboxTimeoutError(sandbox_id="sb-1")
    )

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(vf.SandboxError):
        asyncio.run(mixin.run_background_job(state, command="cmd", timeout=10))
    assert state["sandbox_timeout"] is True


def test_run_background_job_poll_timeout(mixin):
    job = MagicMock()
    not_done = MagicMock(completed=False)
    mixin.sandbox_client.start_background_job = AsyncMock(return_value=job)
    mixin.sandbox_client.get_background_job = AsyncMock(return_value=not_done)

    state = {"sandbox_id": "sb-1"}
    with pytest.raises(CommandTimeoutError):
        asyncio.run(
            mixin.run_background_job(state, command="cmd", timeout=0, poll_interval=1)
        )


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
