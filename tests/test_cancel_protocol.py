"""Tests for the cancel request protocol between ZMQ client and server."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import msgpack
import pytest

from verifiers.workers.client.zmq_env_client import ZMQEnvClient
from verifiers.workers.server.zmq_env_server import ZMQEnvServer
from verifiers.workers.types import (
    CancelRequest,
    HealthRequest,
    HealthResponse,
    PendingRequest,
)


class TestCancelRequestType:
    """Tests for CancelRequest serialization and validation."""

    def test_cancel_request_fields(self):
        req = CancelRequest(cancel_request_ids=["abc123", "def456"])
        assert req.request_type == "cancel"
        assert req.cancel_request_ids == ["abc123", "def456"]

    def test_cancel_request_roundtrip(self):
        req = CancelRequest(cancel_request_ids=["abc123"])
        dumped = req.model_dump(mode="python")
        restored = CancelRequest.model_validate(dumped)
        assert restored.cancel_request_ids == ["abc123"]
        assert restored.request_type == "cancel"

    def test_cancel_request_msgpack_roundtrip(self):
        req = CancelRequest(cancel_request_ids=["a", "b", "c"])
        packed = msgpack.packb(req.model_dump(mode="python"), use_bin_type=True)
        unpacked = msgpack.unpackb(packed, raw=False)
        assert unpacked["request_type"] == "cancel"
        assert unpacked["cancel_request_ids"] == ["a", "b", "c"]


class TestClientSendCancel:
    """Tests for client-side cancel message sending."""

    @pytest.mark.asyncio
    async def test_send_cancel_sends_message(self):
        """send_cancel() sends a properly formatted cancel message."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
        )
        sent_frames = []

        async def capture_send(frames):
            sent_frames.append(frames)

        with patch.object(client.socket, "send_multipart", new=capture_send):
            await client.send_cancel(["req1", "req2"])

        assert len(sent_frames) == 1
        frames = sent_frames[0]
        assert len(frames) == 2  # [cancel_id, payload]

        payload = msgpack.unpackb(frames[1], raw=False)
        assert payload["request_type"] == "cancel"
        assert payload["cancel_request_ids"] == ["req1", "req2"]

        await client.close()

    @pytest.mark.asyncio
    async def test_send_cancel_empty_list_is_noop(self):
        """send_cancel() with empty list does nothing."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
        )

        send_mock = AsyncMock()
        with patch.object(client.socket, "send_multipart", new=send_mock):
            await client.send_cancel([])

        send_mock.assert_not_called()
        await client.close()

    @pytest.mark.asyncio
    async def test_send_cancel_swallows_errors(self):
        """send_cancel() does not raise on send failure."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
        )

        async def fail_send(frames):
            raise RuntimeError("socket closed")

        with patch.object(client.socket, "send_multipart", new=fail_send):
            # Should not raise
            await client.send_cancel(["req1"])

        await client.close()


class TestClientCancelledError:
    """Tests for CancelledError handling in send_request."""

    @pytest.mark.asyncio
    async def test_cancelled_error_cleans_up_and_sends_cancel(self):
        """CancelledError during send_request cleans up pending entry and sends cancel."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
        )

        cancel_ids_sent = []

        async def mock_send_multipart(frames):
            # After send, schedule cancellation of the task
            async def cancel_after():
                await asyncio.sleep(0.05)
                # Cancel the pending future
                async with client.pending_lock:
                    for pending in client.pending_requests.values():
                        if not pending.future.done():
                            pending.future.cancel()

            asyncio.create_task(cancel_after())

        async def capture_send_cancel(request_ids):
            cancel_ids_sent.extend(request_ids)

        with (
            patch.object(client.socket, "connect"),
            patch.object(client.socket, "send_multipart", new=mock_send_multipart),
            patch.object(client, "send_cancel", new=capture_send_cancel),
        ):
            await client.ensure_started()

            with pytest.raises(asyncio.CancelledError):
                await client.send_request(HealthRequest(), HealthResponse, timeout=5.0)

        # Pending request should have been cleaned up
        assert len(client.pending_requests) == 0
        # Cancel should have been sent to server
        assert len(cancel_ids_sent) == 1

        await client.close()


class TestCancelAllPendingSendsCancel:
    """Tests for cancel_all_pending sending cancel messages to the server."""

    @pytest.mark.asyncio
    async def test_cancel_all_pending_sends_cancel_to_server(self):
        """cancel_all_pending() sends a cancel message for all pending request IDs."""
        client = ZMQEnvClient(
            address="tcp://127.0.0.1:5555",
            health_check_interval=0,
        )

        # Add pending requests
        future1 = asyncio.Future()
        future2 = asyncio.Future()
        async with client.pending_lock:
            client.pending_requests["req_aaa"] = PendingRequest(
                request_id="req_aaa",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
                future=future1,
            )
            client.pending_requests["req_bbb"] = PendingRequest(
                request_id="req_bbb",
                request=HealthRequest(),
                submitted_at=time.time(),
                timeout=10.0,
                future=future2,
            )

        cancel_ids_sent = []

        async def capture_send_cancel(request_ids):
            cancel_ids_sent.extend(request_ids)

        with patch.object(client, "send_cancel", new=capture_send_cancel):
            cancelled = await client.cancel_all_pending(
                "test cancel", use_cancelled=True
            )

        assert len(cancelled) == 2
        assert set(cancel_ids_sent) == {"req_aaa", "req_bbb"}

        await client.close()


class TestServerHandleCancel:
    """Tests for server-side cancel handling."""

    @pytest.mark.asyncio
    async def test_handle_cancel_cancels_tracked_task(self):
        """_handle_cancel() cancels tasks tracked in request_tasks."""
        task = asyncio.create_task(asyncio.sleep(100))

        server = ZMQEnvServer.__new__(ZMQEnvServer)
        server.request_tasks = {"req123": task}

        import logging

        server.logger = logging.getLogger("test")

        raw = {"request_type": "cancel", "cancel_request_ids": ["req123"]}
        server._handle_cancel(raw)

        # Task should have cancellation requested
        assert task.cancelling()
        assert "req123" not in server.request_tasks

        # Let the event loop process the cancellation
        with pytest.raises(asyncio.CancelledError):
            await task

    def test_handle_cancel_ignores_unknown_ids(self):
        """_handle_cancel() silently ignores request IDs not in request_tasks."""
        server = ZMQEnvServer.__new__(ZMQEnvServer)
        server.request_tasks = {}

        import logging

        server.logger = logging.getLogger("test")

        raw = {"request_type": "cancel", "cancel_request_ids": ["nonexistent"]}
        # Should not raise
        server._handle_cancel(raw)

    @pytest.mark.asyncio
    async def test_handle_cancel_ignores_already_done_tasks(self):
        """_handle_cancel() does not error on already-completed tasks."""
        future = asyncio.get_running_loop().create_future()
        future.set_result(None)

        server = ZMQEnvServer.__new__(ZMQEnvServer)
        server.request_tasks = {"req_done": future}

        import logging

        server.logger = logging.getLogger("test")

        raw = {"request_type": "cancel", "cancel_request_ids": ["req_done"]}
        server._handle_cancel(raw)

        # Should have been popped from the dict
        assert "req_done" not in server.request_tasks

    def test_handle_cancel_invalid_request(self):
        """_handle_cancel() logs warning on invalid cancel request."""
        server = ZMQEnvServer.__new__(ZMQEnvServer)
        server.request_tasks = {}

        import logging

        server.logger = logging.getLogger("test")

        # Missing required field
        raw = {"request_type": "cancel"}
        # Should not raise
        server._handle_cancel(raw)
