import sys

import pytest
from typing import cast

import verifiers as vf
from tasksets import textarena


class FakeNltk:
    def __init__(self):
        self.downloads: list[tuple[str, bool]] = []

    def download(self, package: str, quiet: bool = False):
        self.downloads.append((package, quiet))


class FakeTextArenaState:
    def __init__(self):
        self.game_state: dict[str, str] = {}
        self.done = False
        self.game_info: dict[int, dict[str, str]] = {}


class FakeTextArenaEnv:
    dictionary = {"words": {"apple", "berry", "cider"}}
    word_list = ["apple", "berry", "cider"]

    def __init__(self, env_id: str = "Wordle-v0"):
        self.env_id = env_id
        self.guesses: list[str] = []
        self.reset_calls = 0
        self.state = FakeTextArenaState()

    def reset(self, num_players: int):
        assert num_players == 1
        self.reset_calls += 1
        self.guesses = []
        self.state = FakeTextArenaState()

    def get_observation(self):
        if not self.guesses:
            return 0, "Guess the word. [GAME] Use <guess>[word]</guess>."
        return 0, "Board [GAME] Feedback:\nmiss\nY----\ntry again"

    def step(self, guess: str):
        self.guesses.append(guess)
        secret = self.state.game_state.get("secret_word")
        if guess == f"[{secret}]":
            self.state.done = True
            self.state.game_info = {0: {"reason": "Solved."}}


class FakeTextArenaModule:
    Env = FakeTextArenaEnv
    State = FakeTextArenaState

    def __init__(self):
        self.envs: list[FakeTextArenaEnv] = []

    def make(self, env_id: str):
        env = FakeTextArenaEnv(env_id=env_id)
        self.envs.append(env)
        return env


@pytest.fixture
def fake_textarena(monkeypatch):
    fake_nltk = FakeNltk()
    fake_ta = FakeTextArenaModule()
    monkeypatch.setitem(sys.modules, textarena.__name__, textarena)
    monkeypatch.setattr(textarena, "nltk", fake_nltk)
    monkeypatch.setattr(textarena, "ta", fake_ta)
    return fake_nltk, fake_ta


def test_textarena_taskset_imports_from_package():
    assert textarena.TextArenaTaskset
    assert textarena.TextArenaTasksetConfig


def test_textarena_taskset_is_generic_over_config_type(fake_textarena):
    class CustomTextArenaConfig(textarena.TextArenaTasksetConfig):
        game: str = "FakeWordle-v0"
        answer_state_key: str = "secret_word"

    class CustomTextArenaTaskset(textarena.TextArenaTaskset[CustomTextArenaConfig]):
        pass

    taskset = CustomTextArenaTaskset(config=CustomTextArenaConfig())

    assert isinstance(taskset.config, CustomTextArenaConfig)


def test_textarena_taskset_builds_train_and_eval_splits(fake_textarena):
    fake_nltk, _ = fake_textarena
    taskset = textarena.TextArenaTaskset(
        config=textarena.TextArenaTasksetConfig(
            game="FakeWordle-v0",
            answer_state_key="secret_word",
            num_train_examples=2,
            num_eval_examples=2,
            seed=1,
        )
    )

    tasks = list(taskset.get_dataset())
    eval_rows = list(taskset.get_eval_dataset())

    assert taskset.config.system_prompt is None
    assert [task["example_id"] for task in tasks] == [0, 1]
    assert [task["example_id"] for task in eval_rows] == [0, 1]
    assert all(task["answer"] in FakeTextArenaEnv.word_list for task in tasks)
    assert all(task["answer"] in FakeTextArenaEnv.word_list for task in eval_rows)
    assert tasks[0]["prompt"] == [
        {
            "role": "user",
            "content": "Guess the word. [GAME] Use <guess>[word]</guess>.",
        }
    ]
    assert fake_nltk.downloads[:2] == [
        ("words", True),
        ("averaged_perceptron_tagger_eng", True),
    ]


def test_textarena_taskset_flattens_dict_word_list(fake_textarena, monkeypatch):
    monkeypatch.setattr(
        FakeTextArenaEnv,
        "word_list",
        {"common": ["apple", "berry"], "rare": "cider"},
    )

    taskset = textarena.TextArenaTaskset(
        config=textarena.TextArenaTasksetConfig(
            game="FakeWordle-v0",
            answer_state_key="secret_word",
            num_train_examples=3,
            num_eval_examples=0,
            seed=1,
        )
    )

    word_list = ["apple", "berry", "cider"]
    assert all(row["answer"] in word_list for row in taskset.get_dataset())


def test_textarena_taskset_loads_user(fake_textarena):
    taskset = textarena.TextArenaTaskset(
        config=textarena.TextArenaTasksetConfig(
            game="FakeWordle-v0",
            answer_state_key="secret_word",
            num_train_examples=1,
            num_eval_examples=0,
        )
    )

    assert isinstance(taskset.user, textarena.TextArenaUser)


@pytest.mark.asyncio
async def test_textarena_user_steps_env_and_stops_when_game_finishes(fake_textarena):
    taskset = textarena.TextArenaTaskset(
        config=textarena.TextArenaTasksetConfig(
            game="FakeWordle-v0",
            answer_state_key="secret_word",
            num_train_examples=1,
            num_eval_examples=0,
        )
    )
    task = taskset.to_task({"example_id": 0, "prompt": [], "answer": "apple"})
    state = vf.State.for_task(task)
    state["completion"] = [
        vf.AssistantMessage(content="I will guess <guess>[apple]</guess>.")
    ]

    env = vf.Env(taskset=taskset, harness=vf.Harness(config=vf.HarnessConfig()))
    messages = await env.harness.runtime.user_messages(task, state)
    ta_env = cast(FakeTextArenaEnv, state["textarena_env"])

    assert ta_env.guesses == ["[apple]"]
    assert ta_env.state.game_state["secret_word"] == "apple"
    assert messages == [{"role": "user", "content": "Solved."}]
    assert state["done"] is True
    assert state["stop_condition"] == "textarena_done"


@pytest.mark.asyncio
async def test_textarena_user_returns_wordle_feedback_for_unfinished_game(
    fake_textarena,
):
    taskset = textarena.TextArenaTaskset(
        config=textarena.TextArenaTasksetConfig(
            game="FakeWordle-v0",
            answer_state_key="secret_word",
            num_train_examples=1,
            num_eval_examples=0,
        )
    )
    task = taskset.to_task({"example_id": 0, "prompt": [], "answer": "apple"})
    state = vf.State.for_task(task)
    state["completion"] = [
        vf.AssistantMessage(content="I will guess <guess>[berry]</guess>.")
    ]

    env = vf.Env(taskset=taskset, harness=vf.Harness(config=vf.HarnessConfig()))
    messages = await env.harness.runtime.user_messages(task, state)
    ta_env = cast(FakeTextArenaEnv, state["textarena_env"])

    assert ta_env.guesses == ["[berry]"]
    assert messages == [
        {
            "role": "user",
            "content": "Board [GAME] Feedback:\nmiss\nY----\ntry again",
        }
    ]
    assert state.get("done") is None
