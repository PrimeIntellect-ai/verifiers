import pytest


def test_wordle_format_observation_extracts_latest_feedback():
    from environments.wordle_v1 import wordle_v1

    taskset = wordle_v1.WordleTaskset.__new__(wordle_v1.WordleTaskset)
    assert (
        taskset.format_observation("intro [GAME] Feedback:\nmiss\nY----\ntry again")
        == "\nmiss\nY----\ntry again"
    )


def test_wordle_load_taskset_requires_wordle_config():
    from environments.wordle_v1 import wordle_v1
    from tasksets.textarena import TextArenaTasksetConfig

    with pytest.raises(AssertionError):
        wordle_v1.load_taskset(
            TextArenaTasksetConfig(game="Wordle-v0", answer_state_key="secret_word")
        )


def test_wordle_taskset_uses_textarena_loaders():
    from environments.wordle_v1 import wordle_v1

    taskset = wordle_v1.WordleTaskset(config=wordle_v1.WordleTasksetConfig())

    assert callable(taskset.load_tasks)
    assert callable(taskset.load_eval_tasks)


def test_wordle_v1_load_taskset_reads_system_prompt_path(tmp_path, monkeypatch):
    from environments.wordle_v1 import wordle_v1

    class CapturingWordleTaskset:
        def __init__(self, config):
            self.config = config

    prompt = "Optimized Wordle prompt.\n\nPreserve exact text.\n"
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    monkeypatch.setattr(wordle_v1, "WordleTaskset", CapturingWordleTaskset)

    taskset = wordle_v1.load_taskset(
        wordle_v1.WordleTasksetConfig(path_to_system_prompt=str(prompt_path))
    )

    assert taskset.config.system_prompt == prompt
    assert taskset.config.game == "Wordle-v0"
    assert taskset.config.answer_state_key == "secret_word"


def test_wordle_v1_load_taskset_rejects_empty_system_prompt_path(tmp_path, monkeypatch):
    from environments.wordle_v1 import wordle_v1

    class CapturingWordleTaskset:
        def __init__(self, config):
            self.config = config

    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(wordle_v1, "WordleTaskset", CapturingWordleTaskset)

    with pytest.raises(ValueError, match="must not be empty"):
        wordle_v1.load_taskset(
            wordle_v1.WordleTasksetConfig(path_to_system_prompt=str(prompt_path))
        )


@pytest.mark.asyncio
async def test_wordle_v1_rewards_match_wordle_protocol():
    from environments.wordle_v1 import wordle_v1
    import verifiers as vf

    taskset = wordle_v1.WordleTaskset.__new__(wordle_v1.WordleTaskset)
    task = vf.Task({"answer": "apple"}).freeze()
    state = vf.State.for_task(task)
    state["completion"] = [
        vf.AssistantMessage(content="<guess>[berry]</guess>"),
        vf.UserMessage(content="miss\nGY---\ntry again"),
        vf.AssistantMessage(content="<guess>[apple]</guess>"),
    ]

    assert await taskset.correct_answer(task, state) == 1.0
    assert await taskset.length_bonus(task, state) == 0.5
    assert await taskset.partial_answer(task, state) == 0.0
    assert await taskset.format_reward(task, state) == 1.0


@pytest.mark.asyncio
async def test_wordle_v1_partial_answer_scans_past_non_guess_messages():
    from environments.wordle_v1 import wordle_v1
    import verifiers as vf

    taskset = wordle_v1.WordleTaskset.__new__(wordle_v1.WordleTaskset)
    task = vf.Task({"answer": "apple"}).freeze()
    state = vf.State.for_task(task)
    state["completion"] = [
        vf.UserMessage(content="miss\nGGGGG\ntry again"),
        vf.AssistantMessage(content="<guess>[apple]</guess>"),
        vf.AssistantMessage(content="I already found it."),
    ]

    assert await taskset.partial_answer(task, state) == 0.0


@pytest.mark.asyncio
async def test_wordle_v1_rewards_treat_missing_completion_as_empty():
    from environments.wordle_v1 import wordle_v1
    import verifiers as vf

    taskset = wordle_v1.WordleTaskset.__new__(wordle_v1.WordleTaskset)
    task = vf.Task({"answer": "apple"}).freeze()
    state = vf.State.for_task(task)

    assert await taskset.correct_answer(task, state) == 0.0
    assert await taskset.length_bonus(task, state) == 0.0
    assert await taskset.partial_answer(task, state) == 0.0
    assert await taskset.format_reward(task, state) == 0.0


def test_wordle_taskset_declares_rewards_as_methods():
    from environments.wordle_v1 import wordle_v1

    for name in ("correct_answer", "partial_answer", "length_bonus", "format_reward"):
        assert getattr(getattr(wordle_v1.WordleTaskset, name), "reward") is True
