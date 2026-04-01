"""Tests for the env server and client.

Covers:
- Health-check state transitions (STARTUP -> HEALTHY -> UNHEALTHY)
- Request retry on ServerError and recovery timeouts
- Server startup waiting
- Cancellation propagation (client -> router -> worker)
"""

import asyncio
import contextlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verifiers.types import ClientConfig, RolloutInput, UserMessage
from verifiers.utils.serve_utils import get_free_port
from verifiers.serve import (
    HealthRequest,
    HealthResponse,
    PendingRequest,
    RunRolloutRequest,
    RunRolloutResponse,
    ServerState,
    ZMQEnvClient,
    ZMQEnvServer,
)


def make_client(address: str = "tcp://127.0.0.1:5555", **kwargs) -> ZMQEnvClient:
    """Create a ZMQEnvClient with health checks disabled by default."""
    kwargs.setdefault("health_check_interval", 0)
    return ZMQEnvClient(address=address, **kwargs)


def make_mock_server(address: str) -> ZMQEnvServer:
    """Create a ZMQEnvServer with a mocked environment (no real env loading)."""
    with patch("verifiers.serve.server.env_server.vf") as mock_vf:
        mock_env = MagicMock()
        mock_env._teardown = AsyncMock()
        mock_vf.load_environment.return_value = mock_env
        mock_vf.setup_logging = MagicMock()
        return ZMQEnvServer(env_id="test", address=address)


def make_rollout_request() -> RunRolloutRequest:
    """Create a minimal RunRolloutRequest for testing."""
    return RunRolloutRequest(
        input=RolloutInput(
            example_id=0,
            task="test",
            prompt=[UserMessage(content="test")],
        ),
        client_config=ClientConfig(),
        model="test-model",
        sampling_args={},
        max_retries=0,
        state_columns=None,
    )


def make_pending_request(
    request_id: str = "test_req",
    timeout: float = 10.0,
) -> tuple[PendingRequest, asyncio.Future]:
    """Create a PendingRequest with its Future for injection into client state."""
    future: asyncio.Future = asyncio.Future()
    pending = PendingRequest(
        request_id=request_id,
        request=HealthRequest(),
        submitted_at=time.time(),
        timeout=timeout,
        future=future,
    )
    return pending, future


@contextlib.asynccontextmanager
async def run_server_and_client():
    """Start a mock ZMQ server and connected client, tearing both down on exit.

    The router's worker spawning is mocked out so no subprocesses are created.
    Instead, dispatch_request/forward_cancel are replaced with AsyncMock so tests can
    observe request routing without needing real workers.
    """
    port = get_free_port()
    address = f"tcp://127.0.0.1:{port}"

    server = make_mock_server(address)

    # Mock out worker lifecycle — we don't want real subprocesses in unit tests
    server.router.start_workers = MagicMock()
    server.router.dispatch_request = AsyncMock()
    server.router.forward_cancel = AsyncMock()

    stop_event = asyncio.Event()
    server_loop = asyncio.create_task(server.serve(stop_event=stop_event))
    await asyncio.sleep(0.1)  # let server bind and start polling

    client = make_client(address=address)

    try:
        yield server, client
    finally:
        stop_event.set()
        await client.close()
        await server.close()
        server_loop.cancel()
        with contextlib.suppress(BaseException):
            await server_loop


@contextlib.contextmanager
def patch_client_socket(client: ZMQEnvClient, send_side_effect=None):
    """Patch connect and send_multipart on a client's ZMQ socket.

    Args:
        client: The client whose socket to patch.
        send_side_effect: Optional async callable used as the
            ``send_multipart`` replacement.
    """
    patches = [patch.object(client.socket, "connect")]
    if send_side_effect is not None:
        patches.append(
            patch.object(client.socket, "send_multipart", new=send_side_effect)
        )
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


class TestStateTransitions:
    """Tests for health-check-driven state transitions (via dedicated thread callbacks)."""

    @pytest.mark.asyncio
    async def test_startup_to_healthy_to_unhealthy(self):
        """Callbacks drive STARTUP -> HEALTHY -> UNHEALTHY via healthy_event."""
        client = make_client()
        client.loop = asyncio.get_running_loop()

        assert client.server_state == ServerState.STARTUP
        assert not client.healthy_event.is_set()

        # STARTUP -> HEALTHY
        client.on_became_healthy(ServerState.STARTUP)
        assert client.server_state == ServerState.HEALTHY
        assert client.healthy_event.is_set()

        # HEALTHY -> UNHEALTHY (after 5 consecutive failures)
        client.on_became_unhealthy(5)
        await asyncio.sleep(0.1)  # let cancel_all_pending run
        assert client.server_state == ServerState.UNHEALTHY
        assert not client.healthy_event.is_set()

        await client.close()

    @pytest.mark.asyncio
    async def test_unhealthy_cancels_pending_with_server_error(self):
        """HEALTHY -> UNHEALTHY transition cancels pending requests with ServerError."""
        client = make_client()
        client.loop = asyncio.get_running_loop()

        # Start in HEALTHY state
        client.server_state = ServerState.HEALTHY
        client.healthy_event.set()

        # Add a pending request
        pending, future = make_pending_request()
        async with client.pending_lock:
            client.pending_requests[pending.request_id] = pending

        # Trigger UNHEALTHY
        client.on_became_unhealthy(5)
        await asyncio.sleep(0.1)  # let cancel_all_pending run

        assert future.done()
        assert len(client.pending_requests) == 0
        with pytest.raises(RuntimeError, match="unhealthy"):
            future.result()

        await client.close()


class TestWaitForServerStartup:
    """Tests for event-based wait_for_server_startup."""

    @pytest.mark.asyncio
    async def test_delayed_startup(self):
        """Startup succeeds when health thread detects server after a delay."""
        client = make_client()
        client.loop = asyncio.get_running_loop()

        # Simulate health thread detecting server after a delay
        async def simulate_health_thread():
            await asyncio.sleep(0.2)
            client.on_became_healthy(ServerState.STARTUP)

        asyncio.create_task(simulate_health_thread())

        with patch_client_socket(client):
            await client.wait_for_server_startup(timeout=3.0)

        assert client.healthy_event.is_set()

        await client.close()

    @pytest.mark.asyncio
    async def test_startup_timeout(self):
        """Startup raises TimeoutError when server never becomes healthy."""
        client = make_client()

        with patch_client_socket(client):
            with pytest.raises(TimeoutError, match="did not become healthy"):
                await client.wait_for_server_startup(timeout=0.5)

        await client.close()


class TestRetryOnServerError:
    """Tests for send_request retry after ServerError."""

    @pytest.mark.asyncio
    async def test_retry_after_recovery(self):
        """ServerError -> wait for healthy_event -> retry succeeds."""
        client = make_client()

        attempt_count = 0

        async def mock_send(frames, **kwargs):
            nonlocal attempt_count
            # Ignore cancel signals (empty payload)
            if len(frames) == 2 and frames[1] == b"":
                return
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

        with patch_client_socket(client, send_side_effect=mock_send):
            await client.ensure_started()
            response = await client.send_request(
                HealthRequest(), HealthResponse, timeout=5.0
            )

            assert attempt_count == 2
            assert response.success

            await client.close()

    @pytest.mark.asyncio
    async def test_recovery_timeout(self):
        """ServerError + no recovery within timeout -> TimeoutError."""
        client = make_client(recovery_timeout=0.5)

        async def mock_send(frames, **kwargs):
            # Ignore cancel signals (empty payload)
            if len(frames) == 2 and frames[1] == b"":
                return

            async def fail():
                await asyncio.sleep(0.05)
                await client.cancel_all_pending("Connection lost")

            asyncio.create_task(fail())

        with patch_client_socket(client, send_side_effect=mock_send):
            await client.ensure_started()

            with pytest.raises(TimeoutError, match="did not recover"):
                await client.send_request(HealthRequest(), HealthResponse, timeout=5.0)

            await client.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_runtime_error(self):
        """Plain RuntimeError propagates immediately without retry."""
        client = make_client()

        attempt_count = 0

        async def mock_send(frames, **kwargs):
            nonlocal attempt_count
            # Ignore cancel signals (empty payload)
            if len(frames) == 2 and frames[1] == b"":
                return
            attempt_count += 1

            async def fail():
                await asyncio.sleep(0.05)
                req_id = list(client.pending_requests.keys())[0]
                pending = client.pending_requests.get(req_id)
                if pending and not pending.future.done():
                    pending.future.set_exception(RuntimeError("Bad request"))

            asyncio.create_task(fail())

        with patch_client_socket(client, send_side_effect=mock_send):
            await client.ensure_started()

            with pytest.raises(RuntimeError, match="Bad request"):
                await client.send_request(HealthRequest(), HealthResponse, timeout=5.0)

            assert attempt_count == 1

            await client.close()


class TestCancelForwarding:
    """Tests that client-side cancellation is forwarded through the router.

    With the multi-process architecture, the ZMQEnvServer receives cancel
    signals from the client and forwards them via ``router.forward_cancel()``.
    These tests verify the server correctly routes cancels to the router.
    """

    @pytest.mark.asyncio
    async def test_cancel_signal_forwarded_to_router(self):
        """Client cancellation sends empty payload, server calls router.forward_cancel."""
        async with run_server_and_client() as (server, client):
            # Send a request
            client_task = asyncio.create_task(
                client.send_request(
                    make_rollout_request(), RunRolloutResponse, timeout=30
                )
            )

            # Wait for dispatch to be called
            await asyncio.sleep(0.3)
            assert server.router.dispatch_request.call_count == 1

            # Cancel on the client side — this sends an empty-payload frame
            client_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await client_task

            # Give the cancel signal time to propagate
            await asyncio.sleep(0.3)

            # The server should have forwarded the cancel to the router
            assert server.router.forward_cancel.call_count == 1
            call_args = server.router.forward_cancel.call_args
            # forward_cancel(request_id, client_id)
            assert isinstance(call_args[0][0], bytes)  # request_id
            assert isinstance(call_args[0][1], bytes)  # client_id

    @pytest.mark.asyncio
    async def test_timeout_sends_cancel_to_router(self):
        """Client timeout sends cancel signal, server calls router.forward_cancel."""
        async with run_server_and_client() as (server, client):
            # Use a short timeout
            with pytest.raises(TimeoutError):
                await client.send_request(
                    make_rollout_request(), RunRolloutResponse, timeout=0.5
                )

            # Give the cancel signal time to propagate
            await asyncio.sleep(0.3)

            # Dispatch should have been called
            assert server.router.dispatch_request.call_count == 1

            # The server should have forwarded the cancel to the router
            assert server.router.forward_cancel.call_count == 1

    @pytest.mark.asyncio
    async def test_dispatch_called_with_correct_frames(self):
        """Requests are dispatched to the router with client_id, request_id, payload."""
        async with run_server_and_client() as (server, client):
            client_task = asyncio.create_task(
                client.send_request(
                    make_rollout_request(), RunRolloutResponse, timeout=30
                )
            )

            await asyncio.sleep(0.3)

            assert server.router.dispatch_request.call_count == 1
            call_args = server.router.dispatch_request.call_args
            client_id, request_id, payload = call_args[0]
            assert isinstance(client_id, bytes)
            assert isinstance(request_id, bytes)
            assert isinstance(payload, bytes)
            assert len(payload) > 0  # non-empty payload = real request

            client_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await client_task
