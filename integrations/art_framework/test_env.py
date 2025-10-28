import json
from pathlib import Path

import verifiers as vf

from .art_framework import load_environment


def _write_tmp_config(tmp_path: Path) -> str:
    cfg = {
        "name": "calc",
        "system_prompt": "Use tools and return via submit_answer.",
        "completion_tool_name": "submit_answer",
        "tools": [
            {
                "name": "add",
                "description": "Add two integers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
                "implementation": "lambda a, b: a + b",
            },
            {
                "name": "submit_answer",
                "description": "Return final answer",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                "implementation": "lambda answer: answer",
            },
        ],
        "examples": [
            {"question": "Add 2 and 3", "answer": "5"},
            {"question": "Add 7 and 3", "answer": "10"},
            {"question": "Add 1 and 9", "answer": "10"},
            {"question": "Add 4 and 4", "answer": "8"},
        ],
    }
    p = tmp_path / "art_task.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_art_tool_conversion_and_parser(tmp_path):
    cfg_path = _write_tmp_config(tmp_path)
    env = load_environment(task_config_path=cfg_path, max_turns=2)
    assert isinstance(env, vf.ToolEnv)
    # smoke test: ensure tools exist and parser extracts from completion tool
    tool_names = [t.__name__ for t in env.tools]  # type: ignore
    assert "add" in tool_names and "submit_answer" in tool_names

    # construct a fake completion that calls submit_answer
    completion = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "submit_answer", "arguments": json.dumps({"answer": "10"})}}
        ]}
    ]
    parsed = env.parser.parse_answer(completion)
    assert parsed == "10"

    # reward should be 1.0 for matching answer
    prompt = [{"role": "user", "content": "What is 7+3?"}]
    rs = env.rubric.score_rollout_sync(prompt=prompt, completion=completion, answer="10", state={})  # type: ignore[attr-defined]
    assert rs.reward == 1.0


