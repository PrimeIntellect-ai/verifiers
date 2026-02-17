"""Tests for environment server crash detection and recovery."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from verifiers.workers.client.zmq_env_client import ZMQEnvClient
from verifiers.workers.types import (
    HealthRequest,
    HealthResponse,
    PendingTaskInfo,
    ServerState,
)


class TestCancelAllPending:
    """Tests for cancel_all_pending functionality."""

    @pytest.mark.asyncio
    async def test_cancel_all_pending_returns_metadata(self):
        """Test that cancel_all_pending returns correct PendingTaskInfo objects."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # Disable health checks
        )

        # Manually add some pending tasks
        request1 = HealthRequest()
        request2 = HealthRequest()

        async with client._pending_lock:
            client.pending["req1"] = asyncio.Future()
            client.pending["req2"] = asyncio.Future()
            client.pending_tasks["req1"] = PendingTaskInfo(
                request_id="req1",
                request=request1,
                submitted_at=time.time(),
                timeout=10.0,
            )
            client.pending_tasks["req2"] = PendingTaskInfo(
                request_id="req2",
                request=request2,
                submitted_at=time.time(),
                timeout=20.0,
            )

        # Cancel all pending
        cancelled_tasks = await client.cancel_all_pending("Test cancellation")

        # Verify return value
        assert len(cancelled_tasks) == 2
        assert all(isinstance(task, PendingTaskInfo) for task in cancelled_tasks)
        assert {task.request_id for task in cancelled_tasks} == {"req1", "req2"}

        # Verify internal state is cleaned up
        assert len(client.pending) == 0
        assert len(client.pending_tasks) == 0

        # Verify futures are cancelled
        for task in cancelled_tasks:
            # The futures should have been failed with RuntimeError
            pass

        await client.close()

    @pytest.mark.asyncio
    async def test_cancel_all_pending_empty(self):
        """Test cancel_all_pending when there are no pending tasks."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # Disable health checks
        )

        cancelled_tasks = await client.cancel_all_pending("Test cancellation")

        assert len(cancelled_tasks) == 0
        assert len(client.pending) == 0
        assert len(client.pending_tasks) == 0

        await client.close()

    @pytest.mark.asyncio
    async def test_cancel_all_pending_fails_futures(self):
        """Test that cancel_all_pending fails all pending futures."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # Disable health checks
        )

        # Create futures
        future1 = asyncio.Future()
        future2 = asyncio.Future()

        async with client._pending_lock:
            client.pending["req1"] = future1
            client.pending["req2"] = future2
            client.pending_tasks["req1"] = PendingTaskInfo(
                request_id="req1",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
            )
            client.pending_tasks["req2"] = PendingTaskInfo(
                request_id="req2",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
            )

        # Cancel all
        await client.cancel_all_pending("Test error")

        # Verify futures are failed
        assert future1.done()
        assert future2.done()
        with pytest.raises(RuntimeError, match="Test error"):
            future1.result()
        with pytest.raises(RuntimeError, match="Test error"):
            future2.result()

        await client.close()


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_disabled(self):
        """Test that health checks don't start when interval=0."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        # Start the receiver (which would normally start health checks)
        await client._start()
        await asyncio.sleep(0.1)

        # Verify health check task is not running
        assert client._health_check_task is None

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_starts_automatically(self):
        """Test that health check task starts on first request."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=5.0,  # 5 seconds
        )

        # Mock the health method to avoid actual network calls
        client.health = AsyncMock(return_value=True)

        # Mock socket operations - send_multipart needs to return a coroutine
        async def mock_send(*args, **kwargs):
            pass

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            # Manually start the client
            await client._start()

            # Simulate sending a request (this should start health check)
            try:
                # This will timeout, but should start health check
                await asyncio.wait_for(
                    client._send_request(HealthRequest(), HealthResponse, timeout=0.1),
                    timeout=0.2,
                )
            except (asyncio.TimeoutError, TimeoutError):
                pass

            # Give it a moment to start the health check task
            await asyncio.sleep(0.1)

            # Verify health check task is running
            assert client._health_check_task is not None
            assert not client._health_check_task.done()

            await client.close()

    @pytest.mark.asyncio
    async def test_health_check_marks_unhealthy_after_failures(self):
        """Test that consecutive health check failures mark server as unhealthy."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0.5,  # Fast checks for testing
            health_check_timeout=0.2,
        )

        # Mock health to fail
        client.health = AsyncMock(side_effect=RuntimeError("Connection failed"))

        # Start the health check loop
        client._health_check_task = asyncio.create_task(client._health_check_loop())

        # Wait for health checks to run and fail
        await asyncio.sleep(1.5)  # Should have 2-3 failed checks

        # Verify state is now unhealthy
        async with client._state_lock:
            assert client._server_state == ServerState.UNHEALTHY

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_recovers_from_unhealthy(self):
        """Test that health check can recover from unhealthy state."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0.5,  # Fast checks for testing
        )

        # Start with unhealthy state
        async with client._state_lock:
            client._server_state = ServerState.UNHEALTHY
            client._failed_health_checks = 3

        # Mock health to succeed
        client.health = AsyncMock(return_value=True)

        # Start the health check loop
        client._health_check_task = asyncio.create_task(client._health_check_loop())

        # Wait for health check to run
        await asyncio.sleep(1.0)

        # Verify state is now healthy
        async with client._state_lock:
            assert client._server_state == ServerState.HEALTHY
            assert client._failed_health_checks == 0

        await client.close()


class TestWaitForRecovery:
    """Tests for wait_for_recovery functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_recovery_success(self):
        """Test that wait_for_recovery succeeds when server recovers."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        # Mock health to succeed
        client.health = AsyncMock(return_value=True)

        # Should return quickly
        await client.wait_for_recovery(timeout=5.0, check_interval=0.5)

        # Verify state is healthy
        async with client._state_lock:
            assert client._server_state == ServerState.HEALTHY

        await client.close()

    @pytest.mark.asyncio
    async def test_wait_for_recovery_timeout(self):
        """Test that wait_for_recovery raises TimeoutError on timeout."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        # Mock health to always fail
        client.health = AsyncMock(side_effect=RuntimeError("Server down"))

        # Should timeout
        with pytest.raises(TimeoutError, match="Server did not recover within"):
            await client.wait_for_recovery(timeout=1.0, check_interval=0.3)

        # Verify state is unhealthy
        async with client._state_lock:
            assert client._server_state == ServerState.UNHEALTHY

        await client.close()

    @pytest.mark.asyncio
    async def test_wait_for_recovery_delayed_success(self):
        """Test recovery after initial failures."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        # Mock health to fail first 2 times, then succeed
        call_count = 0

        async def mock_health(timeout: float | None = 10):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Server down")
            return True

        client.health = mock_health

        # Should eventually succeed
        await client.wait_for_recovery(timeout=3.0, check_interval=0.5)

        # Verify state is healthy
        async with client._state_lock:
            assert client._server_state == ServerState.HEALTHY

        await client.close()


class TestAutoRetry:
    """Tests for automatic retry on server failure."""

    @pytest.mark.asyncio
    async def test_send_request_caches_metadata(self):
        """Test that _send_request caches task metadata."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        request = HealthRequest()

        # Mock socket operations - send_multipart needs to return a coroutine
        async def mock_send(*args, **kwargs):
            pass

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            # Start receiver
            await client._start()

            # Send a request that will timeout
            try:
                await asyncio.wait_for(
                    client._send_request(request, HealthResponse, timeout=0.1),
                    timeout=0.2,
                )
            except (asyncio.TimeoutError, TimeoutError):
                pass

            # The metadata should have been cleaned up on timeout
            assert len(client.pending_tasks) == 0

            await client.close()

    @pytest.mark.asyncio
    async def test_send_request_cleans_metadata_on_success(self):
        """Test that metadata is cleaned up on successful response."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        request = HealthRequest()

        # Mock socket operations - send_multipart needs to return a coroutine
        async def mock_send(*args, **kwargs):
            pass

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client._start()

            # Create a task to send response after a delay
            async def send_response():
                await asyncio.sleep(0.1)
                # Get the request_id from pending
                if client.pending:
                    request_id = list(client.pending.keys())[0]
                    future = client.pending.get(request_id)
                    if future and not future.done():
                        # Simulate a successful response
                        future.set_result({"success": True, "error": None})

            response_task = asyncio.create_task(send_response())

            try:
                response = await asyncio.wait_for(
                    client._send_request(request, HealthResponse, timeout=1.0),
                    timeout=2.0,
                )
                # Verify metadata is cleaned up
                assert len(client.pending_tasks) == 0
                assert response.success
            except asyncio.TimeoutError:
                pass
            finally:
                response_task.cancel()
                try:
                    await response_task
                except asyncio.CancelledError:
                    pass

            await client.close()


class TestCloseCleanup:
    """Tests for client close and cleanup."""

    @pytest.mark.asyncio
    async def test_close_cancels_health_check_task(self):
        """Test that close() cancels the health check task."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=10.0,  # Long interval
        )

        # Mock health to avoid network calls
        client.health = AsyncMock(return_value=True)

        # Mock socket operations - send_multipart needs to return a coroutine
        async def mock_send(*args, **kwargs):
            pass

        # Start the client (which starts health checks)
        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client._start()
            # Manually start health check
            client._health_check_task = asyncio.create_task(client._health_check_loop())
            await asyncio.sleep(0.1)

            # Verify health check is running
            assert client._health_check_task is not None
            assert not client._health_check_task.done()

            # Close the client
            await client.close()

            # Verify health check task is cancelled
            assert client._health_check_task is None

    @pytest.mark.asyncio
    async def test_close_cancels_pending_tasks(self):
        """Test that close() cancels all pending tasks."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        # Add some pending tasks
        async with client._pending_lock:
            client.pending["req1"] = asyncio.Future()
            client.pending_tasks["req1"] = PendingTaskInfo(
                request_id="req1",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
            )

        # Close
        await client.close()

        # Verify cleanup
        assert len(client.pending) == 0
        assert len(client.pending_tasks) == 0
