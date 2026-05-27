from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from gepa.core.adapter import EvaluationBatch

from verifiers.gepa.adapter import (
    VerifiersGEPAAdapter,
    make_reflection_lm,
)
from verifiers.types import ClientConfig


def test_make_reflective_dataset_uses_compact_completion() -> None:
    output: dict[str, Any] = {
        "prompt": [{"role": "user", "content": "Draft the memo."}],
        "completion": [
            {
                "role": "assistant",
                "content": "I drafted it.",
                "tool_calls": [
                    {
                        "name": "read",
                        "arguments": '{"file_path":"source.docx"}',
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "read", "content": "source " * 5000},
        ],
        "answer": "",
        "reward": 0.25,
        "lab_feedback": "Score: 10/40 criteria passed",
    }
    adapter = VerifiersGEPAAdapter(
        env=cast(Any, object()),
        client=cast(Any, object()),
        model="test-model",
        state_columns=["lab_feedback"],
    )
    batch = EvaluationBatch(
        outputs=[output],
        scores=[0.25],
        trajectories=[output],
    )

    dataset = adapter.make_reflective_dataset(
        {"system_prompt": "Current prompt"},
        batch,
        ["system_prompt"],
    )
    record = list(dataset["system_prompt"])[0]

    assert record["query"] == "Draft the memo."
    assert record["completion"] == (
        "Final assistant message:\nI drafted it.\n\n"
        'Tool-call summary:\n- read {"file_path": "source.docx"}'
    )
    assert record["lab_feedback"] == "Score: 10/40 criteria passed"


def test_make_reflection_lm_uses_resolved_prime_headers(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeChatCompletions:
        def create(self, **kwargs: Any) -> Any:
            captured["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeChatCompletions())

    monkeypatch.setenv("PRIME_API_KEY", "prime-key")
    monkeypatch.setenv("PRIME_TEAM_ID", "team-id")
    monkeypatch.setattr("verifiers.gepa.adapter.OpenAI", FakeOpenAI)

    lm = make_reflection_lm(
        ClientConfig(
            api_key_var="PRIME_API_KEY",
            api_base_url="https://api.pinference.ai/api/v1",
            timeout=123,
            connect_timeout=4,
            max_retries=2,
        ),
        "openai/gpt-5.5",
    )

    assert lm("reflect") == "ok"
    assert captured["client"]["api_key"] == "prime-key"
    assert captured["client"]["base_url"] == "https://api.pinference.ai/api/v1"
    assert captured["client"]["max_retries"] == 2
    headers = captured["client"]["http_client"].headers
    assert headers["x-prime-team-id"] == "team-id"
    assert captured["request"]["model"] == "openai/gpt-5.5"
