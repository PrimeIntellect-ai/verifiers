"""Tests for environment server crash detection and recovery."""

import asyncio
import time
from unittest.mock import patch

import msgpack
import pytest

from verifiers.workers.client.zmq_env_client import ZMQEnvClient
from verifiers.workers.types import (
    HealthRequest,
    HealthResponse,
    PendingRequest,
    ServerState,
)


class TestStateTransitions:
    """Tests for health-check-driven state transitions (via dedicated thread callbacks)."""

    @pytest.mark.asyncio
    async def test_startup_to_healthy_to_unhealthy(self):
        """Callbacks drive STARTUP → HEALTHY → UNHEALTHY via healthy_event."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # disable auto thread
        )
        client.loop = asyncio.get_running_loop()

        assert client.server_state == ServerState.STARTUP
        assert not client.healthy_event.is_set()

        # STARTUP → HEALTHY
        client.on_became_healthy(ServerState.STARTUP)
        assert client.server_state == ServerState.HEALTHY
        assert client.healthy_event.is_set()

        # HEALTHY → UNHEALTHY (after 5 consecutive failures)
        client.on_became_unhealthy(5)
        await asyncio.sleep(0.1)  # let _do_cancel_pending run
        assert client.server_state == ServerState.UNHEALTHY
        assert not client.healthy_event.is_set()

        await client.close()

    @pytest.mark.asyncio
    async def test_unhealthy_cancels_pending_with_server_error(self):
        """HEALTHY → UNHEALTHY transition cancels pending requests with ServerError."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # disable auto thread
        )
        client.loop = asyncio.get_running_loop()

        # Start in HEALTHY state
        client.server_state = ServerState.HEALTHY
        client.healthy_event.set()

        # Add a pending request
        future = asyncio.Future()
        async with client.pending_lock:
            client.pending_requests["test_req"] = PendingRequest(
                request_id="test_req",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
                future=future,
            )

        # Trigger UNHEALTHY
        client.on_became_unhealthy(5)
        await asyncio.sleep(0.1)  # let _do_cancel_pending run

        assert future.done()
        assert len(client.pending_requests) == 0
        with pytest.raises(RuntimeError, match="unhealthy"):
            future.result()

        await client.close()


class TestRetryOnServerError:
    """Tests for send_request retry after ServerError."""

    @pytest.mark.asyncio
    async def test_retry_after_recovery(self):
        """ServerError → wait for healthy_event → retry succeeds."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
        )

        attempt_count = 0

        async def mock_send(frames, *args, **kwargs):
            # Ignore cancel messages sent by send_cancel()
            if len(frames) == 2:
                try:
                    payload = msgpack.unpackb(frames[1], raw=False)
                    if payload.get("request_type") == "cancel":
                        return
                except Exception:
                    pass

            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                # First attempt: simulate server crash
                async def fail_then_recover():
                    await asyncio.sleep(0.1)
                    await client.cancel_all_pending("Connection lost")
                    await asyncio.sleep(0.1)
                    client.healthy_event.set()

                asyncio.create_task(fail_then_recover())
            else:
                # Second attempt: succeed
                async def succeed():
                    await asyncio.sleep(0.05)
                    req_id = list(client.pending_requests.keys())[0]
                    pending = client.pending_requests.get(req_id)
                    if pending and not pending.future.done():
                        pending.future.set_result(
                            HealthResponse(success=True).model_dump()
                        )

                asyncio.create_task(succeed())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client.ensure_started()
            response = await client.send_request(
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
                await client.cancel_all_pending("Connection lost")

            asyncio.create_task(fail())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client.ensure_started()

            with pytest.raises(TimeoutError, match="did not recover"):
                await client.send_request(HealthRequest(), HealthResponse, timeout=5.0)

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
                req_id = list(client.pending_requests.keys())[0]
                pending = client.pending_requests.get(req_id)
                if pending and not pending.future.done():
                    pending.future.set_exception(RuntimeError("Bad request"))

            asyncio.create_task(fail())

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send),
        ):
            await client.ensure_started()

            with pytest.raises(RuntimeError, match="Bad request"):
                await client.send_request(HealthRequest(), HealthResponse, timeout=5.0)

            assert attempt_count == 1

            await client.close()


class TestWaitForServerStartup:
    """Tests for event-based wait_for_server_startup."""

    @pytest.mark.asyncio
    async def test_delayed_startup(self):
        """Startup succeeds when health thread detects server after a delay."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # disable auto thread
        )
        client.loop = asyncio.get_running_loop()

        # Simulate health thread detecting server after a delay
        async def simulate_health_thread():
            await asyncio.sleep(0.2)
            client.on_became_healthy(ServerState.STARTUP)

        asyncio.create_task(simulate_health_thread())

        with patch.object(client.socket, "connect"):
            await client.wait_for_server_startup(timeout=3.0)

        assert client.healthy_event.is_set()

        await client.close()

    @pytest.mark.asyncio
    async def test_startup_timeout(self):
        """Startup raises TimeoutError when server never becomes healthy."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,  # disable auto thread
        )

        with patch.object(client.socket, "connect"):
            with pytest.raises(TimeoutError, match="did not become healthy"):
                await client.wait_for_server_startup(timeout=0.5)

        await client.close()
