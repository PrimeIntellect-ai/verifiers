# tests/test_environment_audio_modalities.py
import pytest
from datasets import Dataset

from tests.mock_openai_client import MockOpenAIClient
from verifiers.envs.singleturn_env import SingleTurnEnv

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
    prompt = [
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

    await test_environment.get_model_response(
        client=mock_client,
        model="gpt-4o-audio-preview",
        prompt=prompt,
        oai_tools=None,
        sampling_args=None,
        message_type=None,
    )

    kwargs = mock_client.get_kwargs()
    assert kwargs is not None
    assert kwargs.get("modalities") == ["text"]
    assert kwargs.get("messages") == prompt


@pytest.mark.asyncio
async def test_does_not_override_existing_modalities(mock_client, test_environment):
    prompt = [
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

    await test_environment.get_model_response(
        client=mock_client,
        model="gpt-4o-audio-preview",
        prompt=prompt,
        sampling_args={"modalities": ["text", "audio"]},
        oai_tools=None,
        message_type=None,
    )

    kwargs = mock_client.get_kwargs()
    assert kwargs is not None
    assert kwargs.get("modalities") == ["text", "audio"]


@pytest.mark.asyncio
async def test_does_not_add_modalities_when_no_audio(mock_client, test_environment):
    prompt = [{"role": "user", "content": "hello"}]

    await test_environment.get_model_response(
        client=mock_client,
        model="gpt-4.1-mini",
        prompt=prompt,
        sampling_args=None,
        oai_tools=None,
        message_type=None,
    )

    kwargs = mock_client.get_kwargs()
    assert kwargs is not None
    assert "modalities" not in kwargs
