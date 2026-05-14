import importlib.util
from pathlib import Path

import pytest

import verifiers as vf

pytest.importorskip("chromadb")
pytest.importorskip("datasets")


def _load_wiki_search_module():
    path = Path(__file__).parents[1] / "environments" / "wiki_search" / "wiki_search.py"
    spec = importlib.util.spec_from_file_location("wiki_search_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_wiki_search_judge_accepts_typed_tool_call_messages(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    wiki_search = _load_wiki_search_module()
    env = wiki_search.load_environment(chroma_db_dir=str(tmp_path / "chroma"))
    judge_rubric = env.rubric.rubrics[0]

    async def fake_judge(prompt, completion, answer, state):
        assert completion[0].content is None
        assert completion[-1].content == "The answer is St. Lawrence University."
        return "yes"

    judge_rubric.class_objects["judge"] = fake_judge
    state = {
        "prompt": [{"role": "user", "content": "Which university did Kirk Douglas support?"}],
        "completion": [
            vf.AssistantMessage(content=None),
            vf.AssistantMessage(content="The answer is St. Lawrence University."),
        ],
        "answer": "St. Lawrence University",
        "task": "default",
        "trajectory": [],
    }

    await env.rubric.score_rollout(state)

    assert state["metrics"]["judge_reward_func"] == 1.0
    assert state["reward"] == 1.0
