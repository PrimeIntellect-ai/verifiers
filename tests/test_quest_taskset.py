import pytest
import verifiers as vf
from pydantic import BaseModel, ValidationError

from verifiers.envs.experimental.composable.tasksets.search.quest.taskset import (
    QuestOpenAIClient,
)


class _BinaryResult(BaseModel):
    reasoning: str
    result: bool


class _FakeStructuredCompletions:
    async def parse(self, **kwargs):
        response_format = kwargs["response_format"]
        return response_format.model_validate_json(
            r'{"reasoning": "bad \q escape", "result": true}'
        )


class _FakeChat:
    completions = _FakeStructuredCompletions()


class _FakeBeta:
    chat = _FakeChat()


class _FakeOpenAIClient:
    beta = _FakeBeta()


@pytest.mark.asyncio
async def test_quest_structured_parse_error_becomes_invalid_model_response():
    client = QuestOpenAIClient(client=_FakeOpenAIClient(), model="judge-model")

    with pytest.raises(vf.InvalidModelResponseError) as exc_info:
        await client.async_response(
            messages=[{"role": "user", "content": "judge this"}],
            response_format=_BinaryResult,
        )

    assert "QUEST judge returned invalid structured response" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValidationError)
