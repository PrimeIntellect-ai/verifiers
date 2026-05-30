import sys

import pytest
import verifiers as vf
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


@pytest.mark.asyncio
async def test_textarena_user_stops_without_guess(fake_textarena):
    taskset = textarena.TextArenaTaskset(
        config=textarena.TextArenaTasksetConfig(
            game="FakeWordle-v0",
            answer_state_key="secret_word",
            num_train_examples=1,
            num_eval_examples=0,
        )
    )
    task = taskset.to_task(
        vf.Task(
            {
                "example_id": 0,
                "prompt": [],
                "answer": "apple",
                "textarena": {
                    "game": "FakeWordle-v0",
                    "answer_state_key": "secret_word",
                },
            }
        )
    )
    state = vf.State.for_task(task)
    state["completion"] = [vf.AssistantMessage(content=None, reasoning_content="think")]

    env = vf.Env(taskset=taskset, harness=vf.Harness(config=vf.HarnessConfig()))
    state = await env.harness.setup_state(task, state)
    messages = await env.harness.runtime.user_messages(task, state)
    ta_env = fake_textarena.envs[-1]

    assert ta_env.guesses == []
    assert messages == [{"role": "user", "content": "No <guess>...</guess> tag found."}]
    assert state["stop_condition"] == "textarena_no_guess"
