import pytest


@pytest.mark.asyncio
async def test_wordle_textarena_user_returns_observation(monkeypatch):
    from tasksets import textarena
    from tasksets.textarena import TextArenaTask
    import verifiers.v1 as vf

    class FakeTextArenaState:
        def __init__(self):
            self.game_state: dict[str, str] = {}
            self.done = False
            self.game_info: dict[int, dict[str, str]] = {}

    class FakeTextArenaEnv:
        def __init__(self):
            self.state = FakeTextArenaState()

        def reset(self, num_players: int):
            assert num_players == 1

        def step(self, guess: str):
            assert guess == "[berry]"

        def get_observation(self):
            return 0, "intro [GAME] Feedback:\nmiss\nY----\ntry again"

    class FakeTextArenaModule:
        Env = FakeTextArenaEnv

        def make(self, env_id: str):
            assert env_id == "Wordle-v0"
            return FakeTextArenaEnv()

    monkeypatch.setattr(textarena, "ta", FakeTextArenaModule())
    textarena.SESSION.env = None
    task = TextArenaTask(
        answer="apple",
        textarena={"game": "Wordle-v0", "answer_state_key": "secret_word"},
    )
    state = vf.State(task_id=task.task_id)
    state.add_turn(
        vf.Turn(
            prompt=task.prompt,
            completion=[vf.AssistantMessage(content="<guess>[berry]</guess>")],
        )
    )

    response = textarena.textarena_respond(
        task.to_record(),
        state.model_dump(mode="json", exclude_none=True),
        [],
    )

    assert response["messages"] == [
        {"role": "user", "content": "intro [GAME] Feedback:\nmiss\nY----\ntry again"}
    ]


def test_wordle_load_environment_coerces_taskset_config():
    from environments.wordle_v1.wordle_v1 import taskset as wordle_v1
    from tasksets.textarena import TextArenaTasksetConfig
    import verifiers.v1 as vf
    from verifiers.v1.loaders import load_environment_from_components

    env = load_environment_from_components(
        wordle_v1,
        {
            "config": vf.EnvConfig(
                taskset=TextArenaTasksetConfig(
                    game="Wordle-v0", answer_state_key="secret_word"
                )
            )
        },
    )

    assert isinstance(env.taskset.config, wordle_v1.WordleTasksetConfig)
    assert env.taskset.config.game == "Wordle-v0"
    assert env.taskset.config.answer_state_key == "secret_word"


def test_wordle_taskset_uses_textarena_loaders():
    from environments.wordle_v1.wordle_v1 import taskset as wordle_v1

    taskset = wordle_v1.WordleTaskset(config=wordle_v1.WordleTasksetConfig())

    assert callable(taskset.load_tasks)
    assert taskset.user is not None
    assert taskset.user.server.command is not None


def test_wordle_v1_load_taskset_reads_system_prompt_path(tmp_path):
    from environments.wordle_v1.wordle_v1 import taskset as wordle_v1
    import verifiers.v1 as vf

    prompt = "Optimized Wordle prompt.\n\nPreserve exact text.\n"
    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    taskset = wordle_v1.load_taskset(
        wordle_v1.WordleTasksetConfig(
            system_prompt=vf.SystemPromptConfig(path=str(prompt_path))
        )
    )

    assert taskset.system_prompt == [{"role": "system", "content": prompt}]
    assert taskset.config.game == "Wordle-v0"
    assert taskset.config.answer_state_key == "secret_word"


def test_wordle_v1_load_taskset_rejects_empty_system_prompt_path(tmp_path):
    from environments.wordle_v1.wordle_v1 import taskset as wordle_v1
    import verifiers.v1 as vf

    prompt_path = tmp_path / "system_prompt.txt"
    prompt_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        wordle_v1.load_taskset(
            wordle_v1.WordleTasksetConfig(
                system_prompt=vf.SystemPromptConfig(path=str(prompt_path))
            )
        )


@pytest.mark.asyncio
async def test_wordle_v1_rewards_match_wordle_protocol():
    from environments.wordle_v1.wordle_v1 import taskset as wordle_v1
    from tasksets.textarena import TextArenaTask
    import verifiers.v1 as vf

    taskset = wordle_v1.WordleTaskset(config=wordle_v1.WordleTasksetConfig())
    task = TextArenaTask(
        answer="apple",
        textarena={"game": "Wordle-v0", "answer_state_key": "secret_word"},
    )
    state = vf.State(task_id=task.task_id)
    state.add_turn(
        vf.Turn(
            prompt=task.prompt,
            completion=[
                vf.AssistantMessage(content="<guess>[berry]</guess>"),
                vf.UserMessage(content="miss\nGY---\ntry again"),
                vf.AssistantMessage(content="<guess>[apple]</guess>"),
            ],
        )
    )

    assert await taskset.correct_answer(task, state) == 1.0
    assert await taskset.length_bonus(task, state) == 0.5
    assert await taskset.partial_answer(task, state) == 0.0
    assert await taskset.format_reward(task, state) == 1.0


@pytest.mark.asyncio
async def test_wordle_v1_partial_answer_scans_past_non_guess_messages():
    from environments.wordle_v1.wordle_v1 import taskset as wordle_v1
    from tasksets.textarena import TextArenaTask
    import verifiers.v1 as vf

    taskset = wordle_v1.WordleTaskset(config=wordle_v1.WordleTasksetConfig())
    task = TextArenaTask(
        answer="apple",
        textarena={"game": "Wordle-v0", "answer_state_key": "secret_word"},
    )
    state = vf.State(task_id=task.task_id)
    state.add_turn(
        vf.Turn(
            prompt=task.prompt,
            completion=[
                vf.UserMessage(content="miss\nGGGGG\ntry again"),
                vf.AssistantMessage(content="<guess>[apple]</guess>"),
                vf.AssistantMessage(content="I already found it."),
            ],
        )
    )

    assert await taskset.partial_answer(task, state) == 0.0


@pytest.mark.asyncio
async def test_wordle_v1_rewards_treat_missing_completion_as_empty():
    from environments.wordle_v1.wordle_v1 import taskset as wordle_v1
    from tasksets.textarena import TextArenaTask
    import verifiers.v1 as vf

    taskset = wordle_v1.WordleTaskset(config=wordle_v1.WordleTasksetConfig())
    task = TextArenaTask(
        answer="apple",
        textarena={"game": "Wordle-v0", "answer_state_key": "secret_word"},
    )
    state = vf.State(task_id=task.task_id)

    assert await taskset.correct_answer(task, state) == 0.0
    assert await taskset.length_bonus(task, state) == 0.0
    assert await taskset.partial_answer(task, state) == 0.0
    assert await taskset.format_reward(task, state) == 0.0


def test_wordle_taskset_declares_rewards_as_methods():
    from environments.wordle_v1.wordle_v1 import taskset as wordle_v1

    for name in ("correct_answer", "partial_answer", "length_bonus", "format_reward"):
        assert getattr(getattr(wordle_v1.WordleTaskset, name), "reward") is True
