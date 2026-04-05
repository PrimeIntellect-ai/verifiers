"""Tests for the SandboxManager class."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from verifiers.envs.experimental.resource_managers.sandbox_manager import (
    ManagedSandbox,
    SandboxManager,
)
from verifiers.envs.experimental.resource_managers.base import ResourceState
from verifiers.envs.experimental.resource_managers.retry import RetryConfig
from verifiers.envs.experimental.resource_managers.errors import (
    SandboxCreationError,
    SandboxNotReadyError,
    SandboxOOMError,
)


class TestManagedSandbox:
    """Tests for the ManagedSandbox dataclass."""

    def test_initial_state(self):
        """Test ManagedSandbox initialization."""
        sandbox = ManagedSandbox(id="sandbox-1")

        assert sandbox.id == "sandbox-1"
        assert sandbox.state == ResourceState.CREATING
        assert sandbox.command_times == []
        assert sandbox.ready_wait_time == 0.0

    def test_avg_command_time_empty(self):
        """Test average execution time with no commands."""
        sandbox = ManagedSandbox(id="sandbox-1")
        assert sandbox.avg_command_time == 0.0

    def test_avg_command_time(self):
        """Test average execution time calculation."""
        sandbox = ManagedSandbox(id="sandbox-1")
        sandbox.command_times = [1.0, 2.0, 3.0]

        assert sandbox.avg_command_time == 2.0


@pytest.fixture
def mock_sandbox_client():
    """Create a mock sandbox client."""
    with patch(
        "verifiers.envs.experimental.resource_managers.sandbox_manager.ThreadedAsyncSandboxClient"
    ) as mock_class:
        mock_client = MagicMock()

        # Mock create method with incrementing IDs
        create_counter = [0]
        def create_side_effect(*args, **kwargs):
            create_counter[0] += 1
            mock_sandbox = MagicMock()
            mock_sandbox.id = f"sandbox-{create_counter[0]:03d}"
            return mock_sandbox
        mock_client.create = AsyncMock(side_effect=create_side_effect)

        # Mock delete method
        mock_client.delete = AsyncMock()

        # Mock bulk_delete method
        mock_client.bulk_delete = AsyncMock()

        # Mock wait_for_creation method
        mock_client.wait_for_creation = AsyncMock()

        # Mock execute_command method
        mock_result = MagicMock()
        mock_result.stdout = "ok"  # Return "ok" by default for health checks
        mock_result.stderr = ""
        mock_client.execute_command = AsyncMock(return_value=mock_result)

        # Mock teardown method
        mock_client.teardown = MagicMock()

        mock_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_create_sandbox_request():
    """Create a mock CreateSandboxRequest."""
    with patch(
        "verifiers.envs.experimental.resource_managers.sandbox_manager.CreateSandboxRequest"
    ) as mock_class:
        mock_request = MagicMock()
        mock_class.return_value = mock_request
        yield mock_request


class TestSandboxManager:
    """Tests for the SandboxManager class."""

    @pytest.fixture
    def manager(self, mock_sandbox_client, mock_create_sandbox_request):
        """Create a SandboxManager for testing."""
        manager = SandboxManager(
            default_request=mock_create_sandbox_request,
            retry_config=RetryConfig(max_attempts=2, initial_delay=0.01),
            limits={"command_timeout_seconds": 30, "health_check_interval_seconds": 10.0},
            enable_health_monitoring=False,
        )
        return manager

    @pytest.mark.asyncio
    async def test_acquire_creates_sandbox(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that acquire creates a sandbox via the API."""
        sandbox = await manager.acquire(rollout_id="rollout-1")

        assert sandbox.id == "sandbox-001"  # First sandbox gets ID 001
        assert sandbox.state == ResourceState.READY
        assert sandbox.rollout_id == "rollout-1"
        mock_sandbox_client.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_failure_raises_error(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that creation failures raise SandboxCreationError."""
        mock_sandbox_client.create = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with pytest.raises(SandboxCreationError):
            await manager.acquire(rollout_id="rollout-1")

    @pytest.mark.asyncio
    async def test_release_deletes_sandbox(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that release deletes the sandbox via the API."""
        sandbox = await manager.acquire(rollout_id="rollout-1")
        sandbox_id = sandbox.id

        assert sandbox.state == ResourceState.READY
        await manager.release(sandbox_id)

        # Verify sandbox was marked as destroyed
        assert sandbox.state == ResourceState.DESTROYED

    @pytest.mark.asyncio
    async def test_wait_for_ready(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test waiting for sandbox to be ready."""
        sandbox = await manager.acquire(rollout_id="rollout-1")
        await manager.wait_for_ready(sandbox.id)

        mock_sandbox_client.wait_for_creation.assert_called()
        assert sandbox.ready_wait_time >= 0

    @pytest.mark.asyncio
    async def test_wait_for_ready_failure(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that wait_for_ready failure raises SandboxNotReadyError."""
        mock_sandbox_client.wait_for_creation = AsyncMock(
            side_effect=RuntimeError("Sandbox failed to start")
        )

        sandbox = await manager.acquire(rollout_id="rollout-1")

        with pytest.raises(SandboxNotReadyError):
            await manager.wait_for_ready(sandbox.id)

        # Failure is recorded in failure_info
        assert sandbox.failure_info.not_ready is True

    @pytest.mark.asyncio
    async def test_execute_command(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test executing a command in a sandbox."""
        sandbox = await manager.acquire(rollout_id="rollout-1")

        # Set up a specific output for this command
        mock_result = MagicMock()
        mock_result.stdout = "command output"
        mock_result.stderr = ""
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        output = await manager.execute_command(sandbox.id, "ls -la")

        mock_sandbox_client.execute_command.assert_called()
        assert output == "command output"
        assert len(sandbox.command_times) == 1

    @pytest.mark.asyncio
    async def test_execute_command_with_stderr(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test command output includes stderr."""
        mock_result = MagicMock()
        mock_result.stdout = "stdout output"
        mock_result.stderr = "stderr output"
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        sandbox = await manager.acquire(rollout_id="rollout-1")
        output = await manager.execute_command(sandbox.id, "ls -la")

        assert "stdout output" in output
        assert "stderr" in output
        assert "stderr output" in output

    @pytest.mark.asyncio
    async def test_execute_command_timeout(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test command timeout handling."""
        from prime_sandboxes import CommandTimeoutError

        sandbox = await manager.acquire(rollout_id="rollout-1")

        # Set up the timeout error after acquire
        mock_sandbox_client.execute_command = AsyncMock(
            side_effect=CommandTimeoutError(
                sandbox_id=sandbox.id,
                command="sleep 100",
                timeout=30
            )
        )

        output = await manager.execute_command(sandbox.id, "sleep 100")

        assert "timed out" in output.lower()
        assert len(sandbox.command_times) == 1

    @pytest.mark.asyncio
    async def test_execute_command_unknown_sandbox(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test executing command on unknown sandbox still works (sandbox=None)."""
        mock_result = MagicMock()
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        # Command executes but no timing is recorded since sandbox is unknown
        output = await manager.execute_command("unknown-id", "ls")
        assert output == "output"

    @pytest.mark.asyncio
    async def test_health_check_success(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test successful health check."""
        mock_result = MagicMock()
        mock_result.stdout = "ok"
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        sandbox = await manager.acquire(rollout_id="rollout-1")
        is_healthy = await manager.health_check(sandbox.id)

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test health check with consecutive failures."""
        mock_result = MagicMock()
        mock_result.stdout = "error"
        mock_sandbox_client.execute_command = AsyncMock(return_value=mock_result)

        sandbox = await manager.acquire(rollout_id="rollout-1")

        # Need 3 consecutive failures (default) to be marked unhealthy
        assert await manager.health_check(sandbox.id) is True  # 1st failure
        assert await manager.health_check(sandbox.id) is True  # 2nd failure
        assert await manager.health_check(sandbox.id) is False  # 3rd failure - now unhealthy

    @pytest.mark.asyncio
    async def test_release_all_bulk_deletes(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test release_all uses bulk delete."""
        with patch(
            "verifiers.envs.experimental.resource_managers.sandbox_manager.SandboxClient"
        ) as mock_sync_class:
            mock_sync_client = MagicMock()
            mock_sync_class.return_value = mock_sync_client

            s1 = await manager.acquire(rollout_id="rollout-1")
            s2 = await manager.acquire(rollout_id="rollout-2")

            await manager.release_all()

            mock_sync_client.bulk_delete.assert_called()
            assert s1.state == ResourceState.DESTROYED
            assert s2.state == ResourceState.DESTROYED

    @pytest.mark.asyncio
    async def test_release_all_uses_sync_client(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that release_all uses sync client for shutdown safety."""
        with patch(
            "verifiers.envs.experimental.resource_managers.sandbox_manager.SandboxClient"
        ) as mock_sync_class:
            mock_sync_client = MagicMock()
            mock_sync_class.return_value = mock_sync_client

            await manager.acquire(rollout_id="rollout-1")
            await manager.acquire(rollout_id="rollout-2")

            await manager.release_all()

            # Should use sync client for bulk delete
            mock_sync_client.bulk_delete.assert_called()

    def test_teardown(self, manager: SandboxManager, mock_sandbox_client):
        """Test teardown shuts down the client."""
        manager.teardown()
        mock_sandbox_client.teardown.assert_called_once()


class TestSandboxManagerErrors:
    """Tests for error tracking in SandboxManager."""

    @pytest.fixture
    def manager(self, mock_sandbox_client, mock_create_sandbox_request):
        """Create a SandboxManager for testing."""
        return SandboxManager(
            default_request=mock_create_sandbox_request,
            retry_config=RetryConfig(max_attempts=1, initial_delay=0.01),
            enable_health_monitoring=False,
        )

    @pytest.mark.asyncio
    async def test_creation_error_tracked(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that creation errors are tracked."""
        mock_sandbox_client.create = AsyncMock(
            side_effect=RuntimeError("API error")
        )

        with pytest.raises(SandboxCreationError):
            await manager.acquire(rollout_id="rollout-1")

        errors = manager.get_errors_for_rollout("rollout-1")
        assert len(errors) == 1
        assert errors[0].phase == "create"

    @pytest.mark.asyncio
    async def test_ready_error_tracked(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that ready errors are tracked in failure_info."""
        mock_sandbox_client.wait_for_creation = AsyncMock(
            side_effect=RuntimeError("Failed to start")
        )

        sandbox = await manager.acquire(rollout_id="rollout-1")

        with pytest.raises(SandboxNotReadyError):
            await manager.wait_for_ready(sandbox.id)

        assert sandbox.failure_info.not_ready is True
        assert "Failed to start" in sandbox.failure_info.error_message

    @pytest.mark.asyncio
    async def test_execute_error_tracked(
        self, manager: SandboxManager, mock_sandbox_client
    ):
        """Test that OOM errors are tracked in failure_info."""
        from prime_sandboxes import SandboxOOMError as PrimeOOM

        sandbox = await manager.acquire(rollout_id="rollout-1")

        mock_sandbox_client.execute_command = AsyncMock(
            side_effect=PrimeOOM(sandbox_id=sandbox.id)
        )

        with pytest.raises(SandboxOOMError):
            await manager.execute_command(sandbox.id, "ls")

        assert sandbox.failure_info.oom is True
