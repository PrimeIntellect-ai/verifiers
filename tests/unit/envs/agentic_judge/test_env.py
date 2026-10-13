"""The agentic judge's task carries the solver trace, with hidden reasoning opt-in."""

import json

from judge_traces import full_trace_fixture

from verifiers.v1.envs.agentic_judge import JudgeTaskConfig
from verifiers.v1.envs.agentic_judge.env import TRACE_FILE, JudgeTask


def test_agentic_judge_trace_hidden_reasoning_toggle():
    task = JudgeTask.from_trace(full_trace_fixture(), JudgeTaskConfig())
    record = json.loads(task.files[TRACE_FILE])
    assistant = record["nodes"][1]["message"]
    assert assistant["content"] == "Let me look it up."
    assert assistant["tool_calls"][0]["name"] == "search"
    assert "reasoning_content" not in assistant
    assert "provider_state" not in assistant

    task = JudgeTask.from_trace(
        full_trace_fixture(), JudgeTaskConfig(include_hidden_reasoning=True)
    )
    assistant = json.loads(task.files[TRACE_FILE])["nodes"][1]["message"]
    assert assistant["reasoning_content"] == "SECRET REASONING"
    assert assistant["provider_state"] == [
        {"type": "reasoning", "data": "SECRET STATE"}
    ]
