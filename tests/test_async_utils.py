"""Tests for the async_utils module."""

from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage

from verifiers.utils.async_utils import tqdm_gather_with_metrics


class MockChatCompletion:
    """Mock ChatCompletion with usage."""

    def __init__(self, completion_tokens: int):
        self.usage = MagicMock(spec=CompletionUsage)
        self.usage.completion_tokens = completion_tokens

        self.choices = [MagicMock(spec=Choice)]
        self.choices[0].message = MagicMock(spec=ChatCompletionMessage)
        self.choices[0].message.content = "Hello"
        self.choices[0].message.role = "assistant"


@pytest.mark.asyncio
async def test_tqdm_gather_with_metrics():
    state1 = {"responses": [MockChatCompletion(10)]}
    state2 = {"responses": [MockChatCompletion(14)]}

    async def task1():
        return (0.8, state1)

    async def task2():
        return (0.9, state2)

    with patch("tqdm.asyncio.tqdm") as mock_tqdm:
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tasks = [task1(), task2()]
        results = await tqdm_gather_with_metrics(tasks, total=2, desc="Running eval")

    assert len(results) == 2
    assert results[0] == (0.8, state1)
    assert results[1] == (0.9, state2)

    mock_tqdm.assert_called_once_with(total=2, desc="Running eval")
    assert mock_pbar.update.call_count == 2
    mock_pbar.close.assert_called_once()

    set_desc_calls = mock_pbar.set_description.call_args_list
    assert len(set_desc_calls) == 2

    desc1 = set_desc_calls[0][0][0]
    assert "reward=0.800" in desc1
    assert "completion_length=10" in desc1

    desc2 = set_desc_calls[1][0][0]
    assert "reward=0.850" in desc2
    assert "completion_length=12" in desc2


@pytest.mark.asyncio
async def test_tqdm_gather_with_metrics_missing_usage():
    class MockResponseNoUsage:
        """Mock response without usage field."""

        pass

    state = {"responses": [MockResponseNoUsage()]}

    async def task():
        return (0.7, state)

    with patch("tqdm.asyncio.tqdm") as mock_tqdm:
        mock_pbar = MagicMock()
        mock_tqdm.return_value = mock_pbar

        tasks = [task()]
        results = await tqdm_gather_with_metrics(tasks, total=1, desc="No usage test")

    assert len(results) == 1
    assert results[0] == (0.7, state)
    mock_pbar.update.assert_called_once()
