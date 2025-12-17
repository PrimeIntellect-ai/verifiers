# tests/test_environment_audio_modalities.py
import pytest
from datasets import Dataset

import verifiers as vf
from tests.mock_openai_client import MockOpenAIClient
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.types import RolloutInput

DUMMY_B64 = "ZHVtbXk="


class MockClientWithKwargsCapture(MockOpenAIClient):
    """Mock client that captures kwargs passed to chat.completions.create."""

    def __init__(self):
        super().__init__()
        self._captured_kwargs = None

        async def _wrap_create(**kwargs):
            self._captured_kwargs = kwargs
            return {"ok": True}

        self.chat.completions.create = _wrap_create

    def get_kwargs(self):
        """Get the captured kwargs from the last create call."""
        return self._captured_kwargs


@pytest.fixture
def mock_client():
    return MockClientWithKwargsCapture()


@pytest.fixture
def test_environment():
    dummy_dataset = Dataset.from_dict({"prompt": ["test"]})
    return SingleTurnEnv(dataset=dummy_dataset, message_type="chat")


@pytest.mark.asyncio
async def test_sets_modalities_text_when_audio_and_missing(
    mock_client, test_environment
):
    prompt: vf.Messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": DUMMY_B64, "format": "wav"},
                },
                {"type": "text", "text": "Describe this audio"},
            ],
        }
    ]

    state = await test_environment.init_state(
        input=RolloutInput(example_id=0, task="test", prompt=prompt),
        client=mock_client,
        model="gpt-4o-audio-preview",
    )

    await test_environment.get_model_response(state, prompt)

    kwargs = mock_client.get_kwargs()
    assert kwargs is not None
    assert kwargs.get("modalities") == ["text"]
    assert kwargs.get("messages") == prompt


@pytest.mark.asyncio
async def test_does_not_override_existing_modalities(mock_client, test_environment):
    prompt: vf.Messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": DUMMY_B64, "format": "wav"},
                }
            ],
        }
    ]

    state = await test_environment.init_state(
        input=RolloutInput(example_id=0, task="test", prompt=prompt),
        client=mock_client,
        model="gpt-4o-audio-preview",
        sampling_args={"modalities": ["text", "audio"]},
    )
    await test_environment.get_model_response(state, prompt)

    kwargs = mock_client.get_kwargs()
    assert kwargs is not None
    assert kwargs.get("modalities") == ["text", "audio"]


@pytest.mark.asyncio
async def test_does_not_add_modalities_when_no_audio(mock_client, test_environment):
    prompt: vf.Messages = [{"role": "user", "content": "hello"}]
    state = await test_environment.init_state(
        input=RolloutInput(example_id=0, task="test", prompt=prompt),
        client=mock_client,
        model="gpt-4.1-mini",
    )
    await test_environment.get_model_response(state, prompt)

    kwargs = mock_client.get_kwargs()
    assert kwargs is not None
    assert "modalities" not in kwargs
