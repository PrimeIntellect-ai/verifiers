import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

from verifiers.envs.sandbox_env import SandboxEnv


@pytest.fixture
def sandbox_env():
    """Fixture to create a SandboxEnv instance with mocked dataset."""
    mock_dataset = Dataset.from_dict({"question": ["mock question"], "info": [{}]})

    mock_async_client_patcher = patch("verifiers.envs.sandbox_env.AsyncSandboxClient")
    mock_request_patcher = patch("verifiers.envs.sandbox_env.CreateSandboxRequest")

    mock_async_client = mock_async_client_patcher.start()
    mock_request_patcher.start()

    mock_async_client_instance = MagicMock()
    mock_async_client.return_value = mock_async_client_instance

    try:
        env = SandboxEnv(dataset=mock_dataset, max_retries=1, base_delay=0.1)
        env.logger = MagicMock()
        env.active_sandboxes = {"sandbox1", "sandbox2", "sandbox3"}
        yield env
    finally:
        mock_async_client_patcher.stop()
        mock_request_patcher.stop()


def test_bulk_delete_sandboxes(sandbox_env):
    """Test the bulk_delete_sandboxes method."""
    # Mock the async bulk_delete method on the sandbox_client
    sandbox_env.sandbox_client.bulk_delete = AsyncMock()

    global_ids_to_delete = ["sandbox1", "sandbox3"]
    asyncio.run(sandbox_env.bulk_delete_sandboxes(global_ids_to_delete))

    # Assertions
    sandbox_env.sandbox_client.bulk_delete.assert_called_once_with(global_ids_to_delete)
    sandbox_env.logger.debug.assert_called_once_with(
        f"Bulk deleted sandboxes: {global_ids_to_delete}"
    )
    assert sandbox_env.active_sandboxes == {"sandbox2"}


def test_bulk_delete_sandboxes_failure(sandbox_env):
    """Test the bulk_delete_sandboxes method when an exception occurs."""
    # Mock the async bulk_delete method to raise an exception
    sandbox_env.sandbox_client.bulk_delete = AsyncMock(
        side_effect=Exception("Deletion failed")
    )

    global_ids_to_delete = ["sandbox1", "sandbox3"]
    asyncio.run(sandbox_env.bulk_delete_sandboxes(global_ids_to_delete))

    sandbox_env.logger.error.assert_called_once_with(
        f"Failed to bulk delete sandboxes {global_ids_to_delete}: Deletion failed"
    )
    assert sandbox_env.active_sandboxes == {
        "sandbox1",
        "sandbox2",
        "sandbox3",
    }  # No change
