"""Tests for the unified event system."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from verifiers.types import (
    CompleteEvent,
    EvalEvent,
    LogEvent,
    LogStreamEvent,
    ProgressEvent,
    SaveEvent,
    StartEvent,
)
from verifiers.utils.event_utils import LogStreamFileWriter


class TestEventTypes:
    """Test that event types are correctly defined."""

    def test_start_event_structure(self):
        """Test StartEvent structure."""
        event: StartEvent = {
            "type": "start",
            "total_rollouts": 10,
            "num_examples": 5,
            "rollouts_per_example": 2,
        }
        assert event["type"] == "start"
        assert event["total_rollouts"] == 10
        assert event["num_examples"] == 5
        assert event["rollouts_per_example"] == 2

    def test_progress_event_structure(self):
        """Test ProgressEvent structure."""
        event: ProgressEvent = {
            "type": "progress",
            "all_outputs": [],
            "new_outputs": [],
            "completed_count": 5,
            "total_count": 10,
        }
        assert event["type"] == "progress"
        assert event["completed_count"] == 5
        assert event["total_count"] == 10

    def test_log_event_structure(self):
        """Test LogEvent structure."""
        event: LogEvent = {
            "type": "log",
            "message": "Test message",
            "level": "info",
            "source": "test",
            "timestamp": 123.456,
        }
        assert event["type"] == "log"
        assert event["message"] == "Test message"
        assert event["level"] == "info"

    def test_save_event_structure(self):
        """Test SaveEvent structure."""
        event: SaveEvent = {
            "type": "save",
            "path": Path("/tmp/test.json"),
            "is_intermediate": False,
            "output_count": 10,
        }
        assert event["type"] == "save"
        assert event["is_intermediate"] is False
        assert event["output_count"] == 10

    def test_complete_event_structure(self):
        """Test CompleteEvent structure."""
        event: CompleteEvent = {
            "type": "complete",
            "total_outputs": 10,
            "avg_reward": 0.85,
            "total_time_ms": 1234.5,
        }
        assert event["type"] == "complete"
        assert event["total_outputs"] == 10
        assert event["avg_reward"] == 0.85


class TestLogStreamFileWriter:
    """Test LogStreamFileWriter for #753 log streaming support."""

    @pytest.mark.asyncio
    async def test_writes_log_stream_to_file(self):
        """Test that log stream events are written to files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = LogStreamFileWriter(base_dir)

            # Emit log stream event
            event: LogStreamEvent = {
                "type": "log_stream",
                "stream_id": "test-stream",
                "source": "test",
                "data": "Hello, world!\n",
                "is_stderr": False,
                "file_path": None,
            }

            await writer(event)

            # Check that file was created
            log_file = base_dir / "test-stream.log"
            assert log_file.exists()
            assert log_file.read_text() == "Hello, world!\n"

    @pytest.mark.asyncio
    async def test_appends_to_log_file(self):
        """Test that multiple events append to the same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = LogStreamFileWriter(base_dir)

            # Emit multiple events
            for i in range(3):
                event: LogStreamEvent = {
                    "type": "log_stream",
                    "stream_id": "test-stream",
                    "source": "test",
                    "data": f"Line {i}\n",
                    "is_stderr": False,
                    "file_path": None,
                }
                await writer(event)

            # Check that all lines were written
            log_file = base_dir / "test-stream.log"
            lines = log_file.read_text().splitlines()
            assert lines == ["Line 0", "Line 1", "Line 2"]

    @pytest.mark.asyncio
    async def test_custom_file_path(self):
        """Test that custom file paths are respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = LogStreamFileWriter(base_dir)
            custom_path = base_dir / "custom" / "path" / "test.log"

            # Emit event with custom path
            event: LogStreamEvent = {
                "type": "log_stream",
                "stream_id": "test-stream",
                "source": "test",
                "data": "Custom path test\n",
                "is_stderr": False,
                "file_path": custom_path,
            }

            await writer(event)

            # Check that file was created at custom path
            assert custom_path.exists()
            assert custom_path.read_text() == "Custom path test\n"

    @pytest.mark.asyncio
    async def test_ignores_non_log_stream_events(self):
        """Test that non-log_stream events are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = LogStreamFileWriter(base_dir)

            # Emit non-log_stream event
            event: LogEvent = {
                "type": "log",
                "message": "test",
                "level": "info",
                "source": "test",
                "timestamp": 0.0,
            }

            await writer(event)  # type: ignore

            # Check that no file was created
            assert len(list(base_dir.glob("*.log"))) == 0

    def test_close_all(self):
        """Test that close_all closes all open file handles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            writer = LogStreamFileWriter(base_dir)

            # Open file via event
            event: LogStreamEvent = {
                "type": "log_stream",
                "stream_id": "test-stream",
                "source": "test",
                "data": "test\n",
                "is_stderr": False,
                "file_path": None,
            }

            asyncio.run(writer(event))

            # Verify file is open
            assert "test-stream" in writer.handles
            assert not writer.handles["test-stream"].closed

            # Close all
            writer.close_all()

            # Verify handles are closed and cleared
            assert len(writer.handles) == 0
