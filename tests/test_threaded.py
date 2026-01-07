"""Tests for Threaded in thread_utils module."""

import asyncio
import time
from functools import cached_property

import pytest

from verifiers.utils.thread_utils import Threaded


class MockRoute:
    """Nested class with attribute, sync and async methods."""

    class_name = "mock_route"

    def __init__(self, name: str = "default_route"):
        self.name = name

    def wait(self, duration: float) -> float:
        """Sync method that waits for a duration and returns it."""
        time.sleep(duration)
        return duration

    async def async_wait(self, duration: float) -> float:
        """Async method that waits for a duration and returns it."""
        await asyncio.sleep(duration)
        return duration


class MockClient:
    """A mock client with an attribute, a sync method, and an async method."""

    class_name = "mock_client"

    def __init__(self, name: str = "default"):
        self.instance_name = name

    def wait(self, duration: float) -> float:
        """Sync method that waits for a duration and returns it."""
        time.sleep(duration)
        return duration

    async def async_wait(self, duration: float) -> float:
        """Async method that waits for a duration and returns it."""
        await asyncio.sleep(duration)
        return duration

    @cached_property
    def route(self) -> MockRoute:
        return MockRoute()


class TestThreaded:
    def test_class_attribute_access(self):
        """Test that class attribute can be accessed by the threaded client."""

        threaded_client = Threaded(factory=MockClient, max_workers=1)
        assert threaded_client.class_name == MockClient.class_name

    def test_instance_attribute_access(self):
        """Test that instance attribute can be accessed by the threaded client."""

        threaded_client = Threaded(factory=MockClient, max_workers=1)
        assert threaded_client.instance_name == MockClient().instance_name

    def test_sync_method_call(self):
        """Test that sync method returns the same result as the wrapped client."""
        mock_client = MockClient(name="test")
        threaded_client = Threaded(
            factory=lambda: MockClient(name="test"), max_workers=1
        )

        mock_result = mock_client.wait(0.01)
        threaded_result = threaded_client.wait(0.01)

        assert mock_result == threaded_result == 0.01

    @pytest.mark.asyncio
    async def test_async_method_call(self):
        """Test that async method returns the same result as the wrapped client."""
        mock_client = MockClient()
        threaded_client = Threaded(factory=MockClient, max_workers=1)

        mock_result = await mock_client.async_wait(0.01)
        threaded_result = await threaded_client.async_wait(0.01)

        assert mock_result == threaded_result == 0.01

    def test_nested_class_attribute_access(self):
        """Test class attribute access on nested classes through ChainedProxy."""

        threaded_client = Threaded(factory=MockClient, max_workers=1)
        assert threaded_client.route.class_name == MockClient().route.class_name

    def test_nested_instance_attribute_access(self):
        """Test instance attribute access on nested classes through ChainedProxy."""

        threaded_client = Threaded(factory=MockClient, max_workers=1)
        assert threaded_client.route.name == MockClient().route.name

    def test_nested_instance_sync_method_call(self):
        """Test sync method call on nested class through ChainedProxy."""
        threaded_client = Threaded(factory=MockClient, max_workers=1)
        assert threaded_client.route.wait(0.01) == 0.01

    @pytest.mark.asyncio
    async def test_nested_instance_async_method_call(self):
        """Test async method call on nested class through ChainedProxy."""
        threaded_client = Threaded(factory=MockClient, max_workers=1)
        assert await threaded_client.route.async_wait(0.01) == 0.01

    @pytest.mark.asyncio
    async def test_async_worker_execution(self):
        """Test that async calls submitted to a single worker run concurrently."""

        num_calls = 10
        wait_duration = 0.1
        threaded_client = Threaded(factory=MockClient, max_workers=1)

        start = time.perf_counter()
        tasks = [threaded_client.async_wait(wait_duration) for _ in range(num_calls)]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

        assert all(r == wait_duration for r in results)
        assert elapsed < 0.5, (
            f"Expected ~0.1s but took {elapsed:.2f}s (should be concurrent)"
        )
        assert elapsed >= wait_duration * 0.9, f"Elapsed {elapsed:.2f}s seems too fast"

    @pytest.mark.asyncio
    async def test_sequential_sync_execution_across_workers(self):
        """Test that sync method calls are distributed across workers for parallel execution."""

        num_calls = 2
        wait_duration = 0.1
        threaded_client = Threaded(factory=MockClient, max_workers=num_calls)

        start = time.perf_counter()
        results = [threaded_client.wait(wait_duration) for _ in range(num_calls)]
        elapsed = time.perf_counter() - start

        assert all(r == wait_duration for r in results)
        assert elapsed > 0.1, (
            f"Expected ~0.1s but took {elapsed:.2f}s (should be sequential)"
        )
