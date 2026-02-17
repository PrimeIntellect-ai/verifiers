"""Tests for environment server crash detection and recovery."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from verifiers.workers.client.zmq_env_client import ZMQEnvClient
from verifiers.workers.types import (
    HealthRequest,
    HealthResponse,
    PendingRequest,
)


class TestCancelAllPending:
    """Tests for cancel_all_pending functionality."""

    @pytest.mark.asyncio
    async def test_cancel_all_pending_returns_metadata(self):
        """Test that cancel_all_pending returns correct PendingRequest objects."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # Disable health checks
        )

        # Manually add some pending tasks
        request1 = HealthRequest()
        request2 = HealthRequest()
        future1 = asyncio.Future()
        future2 = asyncio.Future()

        async with client._pending_lock:
            client.pending_requests["req1"] = PendingRequest(
                request_id="req1",
                request=request1,
                submitted_at=time.time(),
                timeout=10.0,
                future=future1,
            )
            client.pending_requests["req2"] = PendingRequest(
                request_id="req2",
                request=request2,
                submitted_at=time.time(),
                timeout=20.0,
                future=future2,
            )

        # Cancel all pending
        cancelled_requests = await client.cancel_all_pending()

        # Verify return value
        assert len(cancelled_requests) == 2
        assert all(isinstance(req, PendingRequest) for req in cancelled_requests)
        assert {req.request_id for req in cancelled_requests} == {"req1", "req2"}

        # Verify internal state is cleaned up
        assert len(client.pending_requests) == 0

        # Verify futures are cancelled
        for req in cancelled_requests:
            assert req.future.done()
            # The futures should have been failed with RuntimeError
            with pytest.raises(RuntimeError):
                req.future.result()

        await client.close()

    @pytest.mark.asyncio
    async def test_cancel_all_pending_empty(self):
        """Test cancel_all_pending when there are no pending tasks."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # Disable health checks
        )

        cancelled_requests = await client.cancel_all_pending()

        assert len(cancelled_requests) == 0
        assert len(client.pending_requests) == 0

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
            client.pending_requests["req1"] = PendingRequest(
                request_id="req1",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
                future=future1,
            )
            client.pending_requests["req2"] = PendingRequest(
                request_id="req2",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
                future=future2,
            )

        # Cancel all
        await client.cancel_all_pending()

        # Verify futures are failed
        assert future1.done()
        assert future2.done()
        with pytest.raises(RuntimeError):
            future1.result()
        with pytest.raises(RuntimeError):
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
    async def test_health_check_detects_failures(self):
        """Test that consecutive health check failures are detected."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0.5,  # Fast checks for testing
        )

        # Mock health to fail
        client.health = AsyncMock(side_effect=RuntimeError("Connection failed"))

        # Start the health check loop
        client._health_check_task = asyncio.create_task(client._health_check_loop())

        # Wait for health checks to run and fail
        await asyncio.sleep(1.5)  # Should have 2-3 failed checks

        # Verify failed health checks are being tracked
        assert client._failed_health_checks >= 2

        await client.close()

    @pytest.mark.asyncio
    async def test_health_check_recovers_after_failures(self):
        """Test that health check can recover after failures."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0.5,  # Fast checks for testing
        )

        # Start with failed health checks
        client._failed_health_checks = 3

        # Mock health to succeed
        client.health = AsyncMock(return_value=True)

        # Start the health check loop
        client._health_check_task = asyncio.create_task(client._health_check_loop())

        # Wait for health check to run
        await asyncio.sleep(1.0)

        # Verify health checks are now succeeding
        assert client._failed_health_checks == 0

        await client.close()


class TestWaitForServerHealth:
    """Tests for wait_for_server_startup and wait_for_server_recovery functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_server_health_success(self):
        """Test that wait_for_server_startup succeeds when server is healthy."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        # Mock health to succeed
        client.health = AsyncMock(return_value=True)

        # Should return quickly
        await client.wait_for_server_startup(timeout=5.0, interval=0.5)

        # Verify failed health checks reset
        assert client._failed_health_checks == 0

        await client.close()

    @pytest.mark.asyncio
    async def test_wait_for_server_health_timeout(self):
        """Test that wait_for_server_startup raises TimeoutError on timeout."""
        client = ZMQEnvClient(address="tcp://127.0.0.1:5555", health_check_interval=0)

        # Mock health to always fail
        client.health = AsyncMock(side_effect=RuntimeError("Server down"))

        # Should timeout
        with pytest.raises(TimeoutError, match="Server did not become healthy within"):
            await client.wait_for_server_startup(timeout=1.0, interval=0.3)

        await client.close()

    @pytest.mark.asyncio
    async def test_wait_for_server_health_delayed_success(self):
        """Test health check after initial failures."""
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
        await client.wait_for_server_startup(timeout=3.0, interval=0.5)

        # Verify failed health checks reset
        assert client._failed_health_checks == 0

        await client.close()


class TestRequestMetadata:
    """Tests for request metadata caching."""

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
            assert len(client.pending_requests) == 0

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
                if client.pending_requests:
                    request_id = list(client.pending_requests.keys())[0]
                    pending_req = client.pending_requests.get(request_id)
                    if pending_req and not pending_req.future.done():
                        # Simulate a successful response
                        pending_req.future.set_result({"success": True, "error": None})

            response_task = asyncio.create_task(send_response())

            try:
                response = await asyncio.wait_for(
                    client._send_request(request, HealthResponse, timeout=1.0),
                    timeout=2.0,
                )
                # Verify metadata is cleaned up
                assert len(client.pending_requests) == 0
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
            client.pending_requests["req1"] = PendingRequest(
                request_id="req1",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
                future=asyncio.Future(),
            )

        # Close
        await client.close()

        # Verify cleanup
        assert len(client.pending_requests) == 0


class TestAutomaticRetry:
    """Tests for automatic retry after server failure."""

    @pytest.mark.asyncio
    async def test_retry_on_server_failure(self):
        """Test that requests are automatically retried after server failure."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
            max_auto_retries=2,
        )

        request = HealthRequest()
        attempt_count = 0

        # Mock wait_for_server_recovery to succeed immediately
        async def mock_recovery(timeout=None, interval=None):
            pass

        client.wait_for_server_recovery = mock_recovery

        # Mock socket operations
        async def mock_send(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            # Simulate server failure on first attempt
            if attempt_count == 1:
                # Trigger cancellation after a short delay (simulating server death)
                async def fail_request():
                    await asyncio.sleep(0.1)
                    await client.cancel_all_pending("ZMQ socket error: Connection lost")

                asyncio.create_task(fail_request())
            else:
                # Second attempt succeeds - return response immediately
                async def succeed_request():
                    await asyncio.sleep(0.05)
                    request_id = list(client.pending_requests.keys())[0]
                    pending = client.pending_requests.get(request_id)
                    if pending and not pending.future.done():
                        pending.future.set_result({"success": True, "error": None})

                asyncio.create_task(succeed_request())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client._start()

            # Send request - should retry and succeed
            response = await client._send_request(request, HealthResponse, timeout=5.0)

            # Verify retry happened
            assert attempt_count == 2
            assert response.success

            await client.close()

    @pytest.mark.asyncio
    async def test_retry_respects_max_retries(self):
        """Test that retry stops after max_auto_retries is reached."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
            max_auto_retries=2,
        )

        request = HealthRequest()
        attempt_count = 0

        # Mock wait_for_server_recovery to succeed immediately
        async def mock_recovery(timeout=None, interval=None):
            pass

        client.wait_for_server_recovery = mock_recovery

        # Mock socket operations - always fail
        async def mock_send(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            # Always simulate server failure
            async def fail_request():
                await asyncio.sleep(0.05)
                await client.cancel_all_pending("ZMQ socket error: Connection lost")

            asyncio.create_task(fail_request())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client._start()

            # Send request - should retry max_auto_retries times then fail
            with pytest.raises(RuntimeError, match="ZMQ socket error"):
                await client._send_request(request, HealthResponse, timeout=5.0)

            # Verify it tried initial + 2 retries = 3 attempts
            assert attempt_count == 3

            await client.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_non_server_error(self):
        """Test that non-server errors don't trigger retry."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
            max_auto_retries=3,
        )

        request = HealthRequest()
        attempt_count = 0

        # Mock socket operations
        async def mock_send(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            # Simulate a non-server error (e.g., validation error)
            async def fail_with_validation_error():
                await asyncio.sleep(0.05)
                request_id = list(client.pending_requests.keys())[0]
                pending = client.pending_requests.get(request_id)
                if pending and not pending.future.done():
                    pending.future.set_exception(RuntimeError("Invalid request format"))

            asyncio.create_task(fail_with_validation_error())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client._start()

            # Send request - should NOT retry
            with pytest.raises(RuntimeError, match="Invalid request format"):
                await client._send_request(request, HealthResponse, timeout=5.0)

            # Verify no retry (only 1 attempt)
            assert attempt_count == 1

            await client.close()
