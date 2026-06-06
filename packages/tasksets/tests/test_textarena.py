import sys

import pytest
import verifiers.v1 as vf
from tasksets import textarena


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
    fake_ta = FakeTextArenaModule()
    monkeypatch.setitem(sys.modules, textarena.__name__, textarena)
    monkeypatch.setattr(textarena, "ta", fake_ta)
    return fake_ta


def test_textarena_user_steps_empty_guess_when_guess_tag_missing(fake_textarena):
    taskset = textarena.TextArenaTaskset(
        config=textarena.TextArenaTasksetConfig(
            game="FakeWordle-v0",
            answer_state_key="secret_word",
            num_train_examples=1,
            num_eval_examples=0,
        )
    )
    task = taskset.to_task(
        textarena.TextArenaTask(
            row_id=0,
            prompt=[],
            answer="apple",
            textarena={
                "game": "FakeWordle-v0",
                "answer_state_key": "secret_word",
            },
        )
    )
    textarena.SESSION = textarena.TextArenaSession()
    completion = [
        vf.AssistantMessage(content=None, reasoning_content="think").model_dump(
            mode="json", exclude_none=True
        )
    ]
    payload = textarena.textarena_respond(
        task.to_record(), {"completion": completion}, []
    )
    ta_env = fake_textarena.envs[-1]

    assert ta_env.guesses == [""]
    assert payload["messages"] == [
        {
            "role": "user",
            "content": "Board [GAME] Feedback:\nmiss\nY----\ntry again",
        }
    ]
    assert "stop_condition" not in payload
