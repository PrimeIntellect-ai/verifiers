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
    ServerState,
)


class TestStateTransitions:
    """Tests for health-check-driven state transitions."""

    @pytest.mark.asyncio
    async def test_startup_to_healthy_to_unhealthy(self):
        """Health loop drives STARTUP → HEALTHY → UNHEALTHY via _healthy_event."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0.2,
        )

        assert client._server_state == ServerState.STARTUP
        assert not client._healthy_event.is_set()

        # Health succeeds → HEALTHY, event set
        client.health = AsyncMock(return_value=True)
        client._health_check_task = asyncio.create_task(client._health_check_loop())

        await asyncio.sleep(0.3)
        assert client._server_state == ServerState.HEALTHY
        assert client._healthy_event.is_set()

        # Health fails → after 3 consecutive failures → UNHEALTHY, event cleared
        client.health = AsyncMock(return_value=False)

        await asyncio.sleep(0.7)  # 3+ checks
        assert client._server_state == ServerState.UNHEALTHY
        assert not client._healthy_event.is_set()

        await client.close()

    @pytest.mark.asyncio
    async def test_unhealthy_cancels_pending_with_server_error(self):
        """HEALTHY → UNHEALTHY transition cancels pending requests with ServerError."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0.2,
        )

        # Start in HEALTHY state
        client._server_state = ServerState.HEALTHY
        client._healthy_event.set()
        client.health = AsyncMock(return_value=False)

        # Add a pending request
        future = asyncio.Future()
        async with client._pending_lock:
            client._pending_requests["test_req"] = PendingRequest(
                request_id="test_req",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
                future=future,
            )

        client._health_check_task = asyncio.create_task(client._health_check_loop())
        await asyncio.sleep(0.8)  # 3+ failures

        assert future.done()
        assert len(client._pending_requests) == 0
        with pytest.raises(RuntimeError, match="unhealthy"):
            future.result()

        await client.close()


class TestRetryOnServerError:
    """Tests for _send_request retry after ServerError."""

    @pytest.mark.asyncio
    async def test_retry_after_recovery(self):
        """ServerError → wait for _healthy_event → retry succeeds."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
        )

        attempt_count = 0

        async def mock_send(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt: simulate server crash
                async def fail_then_recover():
                    await asyncio.sleep(0.1)
                    await client._cancel_all_pending("Connection lost")
                    await asyncio.sleep(0.1)
                    client._healthy_event.set()

                asyncio.create_task(fail_then_recover())
            else:
                # Second attempt: succeed
                async def succeed():
                    await asyncio.sleep(0.05)
                    req_id = list(client._pending_requests.keys())[0]
                    pending = client._pending_requests.get(req_id)
                    if pending and not pending.future.done():
                        pending.future.set_result(
                            HealthResponse(success=True).model_dump()
                        )

                asyncio.create_task(succeed())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client._ensure_started()
            response = await client._send_request(
                HealthRequest(), HealthResponse, timeout=5.0
            )

            assert attempt_count == 2
            assert response.success

            await client.close()

    @pytest.mark.asyncio
    async def test_recovery_timeout(self):
        """ServerError + no recovery within timeout → TimeoutError."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
            recovery_timeout=0.5,
        )

        async def mock_send(*args, **kwargs):
            async def fail():
                await asyncio.sleep(0.05)
                await client._cancel_all_pending("Connection lost")

            asyncio.create_task(fail())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client._ensure_started()

            with pytest.raises(TimeoutError, match="did not recover"):
                await client._send_request(HealthRequest(), HealthResponse, timeout=5.0)

            await client.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_runtime_error(self):
        """Plain RuntimeError propagates immediately without retry."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
        )

        attempt_count = 0

        async def mock_send(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1

            async def fail():
                await asyncio.sleep(0.05)
                req_id = list(client._pending_requests.keys())[0]
                pending = client._pending_requests.get(req_id)
                if pending and not pending.future.done():
                    pending.future.set_exception(RuntimeError("Bad request"))

            asyncio.create_task(fail())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client._ensure_started()

            with pytest.raises(RuntimeError, match="Bad request"):
                await client._send_request(HealthRequest(), HealthResponse, timeout=5.0)

            assert attempt_count == 1

            await client.close()


class TestWaitForServerStartup:
    """Tests for event-based wait_for_server_startup."""

    @pytest.mark.asyncio
    async def test_delayed_startup(self):
        """Startup succeeds after initial health failures."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0.1,
        )

        call_count = 0

        async def mock_health(timeout: float | None = 10):
            nonlocal call_count
            call_count += 1
            return call_count > 2

        client.health = mock_health

        with patch.object(client.socket, "connect"):
            await client.wait_for_server_startup(timeout=3.0)

        assert client._healthy_event.is_set()

        await client.close()

    @pytest.mark.asyncio
    async def test_startup_timeout(self):
        """Startup raises TimeoutError when server never becomes healthy."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0.1,
        )

        client.health = AsyncMock(return_value=False)

        with patch.object(client.socket, "connect"):
            with pytest.raises(TimeoutError, match="did not become healthy"):
                await client.wait_for_server_startup(timeout=1.0)

        await client.close()
