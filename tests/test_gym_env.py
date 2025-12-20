from __future__ import annotations

import re
from typing import Any

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion import Completion as OAICompletion
from openai.types.completion_choice import CompletionChoice

from verifiers.envs.experimental.gym_env import EpisodicSumRubric, GymEnv


# ----------------- Toy Environment -----------------
class ToyEnv:
    """
    Simple counter environment for testing.
    Observation is "x=<int>". Action is 0 or 1 (delta to add).
    Episode ends when x >= target or max_steps reached.
    Reward is 1.0 when target is reached, else 0.0.
    """

    def __init__(self, start: int = 0, target: int = 3, max_steps: int = 20, **kwargs):
        self.start = int(start)
        self.target = int(target)
        self.max_steps = int(max_steps)
        self.x = self.start
        self.steps = 0
        self.done = False

    def reset(self, **kwargs):
        self.x = self.start
        self.steps = 0
        self.done = False
        return f"x={self.x}", {"target": self.target, "start": self.start}

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode finished, call reset().")
        self.steps += 1
        self.x += int(action)
        done = self.x >= self.target or self.steps >= self.max_steps
        self.done = done
        reward = 1.0 if self.x >= self.target else 0.0
        info = {"x": self.x, "target": self.target, "reached": self.x >= self.target}
        return f"x={self.x}", reward, done, False, info


# ----------------- Mock OpenAI Client -----------------
class _MockChatCompletions:
    async def create(self, *, model: str, messages: list[dict[str, str]], **kwargs):
        last_user = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        n = 0
        m = re.search(r"x\s*=\s*(-?\d+)", str(last_user))
        if m:
            n = int(m.group(1))

        action = "1" if n < 3 else "0"

        message = ChatCompletionMessage.model_construct(
            role="assistant", content=action, tool_calls=None
        )
        choice = Choice.model_construct(
            index=0, message=message, finish_reason="stop", logprobs=None
        )
        return ChatCompletion.model_construct(
            id="mock-chatcmpl",
            choices=[choice],
            created=0,
            model=model,
            object="chat.completion",
            usage=None,
        )


class _MockCompletions:
    async def create(self, *, model: str, prompt: str, **kwargs):
        n = 0
        m = re.search(r"x\s*=\s*(-?\d+)", prompt or "")
        if m:
            n = int(m.group(1))
        action = "1" if n < 3 else "0"

        choice = CompletionChoice.model_construct(
            index=0, text=action, logprobs=None, finish_reason="stop"
        )
        return OAICompletion.model_construct(
            id="mock-cmpl",
            choices=[choice],
            created=0,
            model=model,
            object="text_completion",
            usage=None,
        )


class _MockChat:
    def __init__(self):
        self.completions = _MockChatCompletions()


class MockAsyncOpenAI:
    def __init__(self):
        self.chat = _MockChat()
        self.completions = _MockCompletions()
        self.base_url = "mock://local"


def parse_action(txt: str) -> int:
    m = re.search(r"[-+]?\d+", txt)
    if not m:
        raise ValueError(f"No int in: {txt!r}")
    return 1 if int(m.group(0)) > 0 else 0


@pytest.fixture
def toy_env_class():
    return ToyEnv


@pytest.fixture
def client():
    return MockAsyncOpenAI()


# ----------------- Tests -----------------


def test_basic_rollout_and_reward_sum(toy_env_class, client):
    """Basic rollout reaches target and sums rewards correctly."""
    env = GymEnv(
        env_cls=toy_env_class,
        env_kwargs={"start": 0, "target": 3, "max_steps": 10},
        action_parser=parse_action,
        message_type="chat",
        max_episode_steps=10,
        rubric=EpisodicSumRubric(),
        num_train_episodes=0,
        num_eval_episodes=1,
    )

    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("trajectory", [])

    assert len(steps) > 0
    assert res["reward"] == [1.0]
    assert st.get("gym_done") is True

    last_prompt = steps[-1]["prompt"]
    assert "Episode already ended." in str(last_prompt)


def test_action_parse_error_ends_episode(toy_env_class, client):
    """Action parsing errors end the episode with error feedback."""

    def bad_parser(_txt: str) -> int:
        raise ValueError("no action")

    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=bad_parser,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
    )

    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("trajectory", [])

    assert st.get("gym_done") is True
    last_prompt = steps[-1]["prompt"]
    assert "Action Parsing Error" in str(last_prompt)


def test_max_episode_steps_limits_turns(client):
    """max_episode_steps limits turns even if env never terminates."""

    class NoTermEnv:
        def reset(self, **kwargs):
            return "x=0", {}

        def step(self, action: int):
            return "x=1", 0.0, False, False, {}

    env = GymEnv(
        env_cls=NoTermEnv,
        action_parser=parse_action,
        message_type="chat",
        max_episode_steps=3,
        num_train_episodes=0,
        num_eval_episodes=1,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("trajectory", [])

    assert len(steps) == 3
    assert st.get("gym_done") is False
    assert st.get("is_completed") is True

    last_prompt = steps[-1]["prompt"]
    assert "Episode already ended." not in str(last_prompt)


def test_system_prompt_and_few_shot(toy_env_class, client):
    """System prompt and few-shot examples are included in first prompt."""
    few = [
        {"role": "user", "content": "demo Q"},
        {"role": "assistant", "content": "demo A"},
    ]

    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        system_prompt="SYS",
        few_shot=few,
        num_train_episodes=0,
        num_eval_episodes=1,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    first_prompt = st["prompt"]

    roles = [m["role"] for m in first_prompt]
    contents = [m.get("content") for m in first_prompt]
    assert roles[:4] == ["system", "user", "assistant", "user"]
    assert contents[0] == "SYS"
    assert contents[-1].startswith("x=0")


def test_four_tuple_step_normalization(client):
    """Environments using old 4-tuple step API are normalized to 5-tuple."""

    class FourTupleEnv:
        def reset(self, **kwargs):
            return "x=0", {}

        def step(self, action: int):
            return "x=1", 1.0, True, {"info": "done"}

    env = GymEnv(
        env_cls=FourTupleEnv,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("trajectory", [])

    assert steps[0]["extras"]["gym_info"] == {"info": "done"}
    assert st.get("gym_done") is True


def test_env_kwargs_passed_to_env(toy_env_class, client):
    """env_kwargs are passed to environment constructor."""
    env = GymEnv(
        env_cls=toy_env_class,
        env_kwargs={"start": 5, "target": 10},
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]

    first_obs_msg = st["prompt"][-1]["content"]
    assert first_obs_msg == "x=5"


def test_custom_obs_to_text(client):
    """Custom obs_to_text function or subclass method is used."""

    class NumObsEnv:
        def reset(self, **kwargs):
            return 0, {}

        def step(self, action: int):
            return 1, 0.0, True, False, {}

    class FmtGymEnv(GymEnv):
        def obs_to_text(self, obs: Any) -> str:
            return f"obs_is_{obs}"

    env = FmtGymEnv(
        env_cls=NumObsEnv,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=0,
        num_eval_episodes=1,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    assert st["prompt"][-1]["content"] == "obs_is_0"


def test_missing_env_cls_raises_error():
    """GymEnv requires env_cls argument."""
    with pytest.raises(TypeError):
        GymEnv(action_parser=parse_action)  # type: ignore[call-arg]


def test_completion_mode(toy_env_class, client):
    """Completion mode uses string prompts instead of chat messages."""
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="completion",
        num_train_episodes=0,
        num_eval_episodes=1,
    )

    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]

    assert isinstance(st["prompt"], str)
    assert st["prompt"] == "x=0"
    comp = st.get("completion", "")
    assert isinstance(comp, str)


def test_dataset_generation_chat_mode(toy_env_class):
    """gym_to_hf generates datasets with question column for chat mode."""
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=10,
        num_eval_episodes=3,
    )

    assert env.dataset is not None
    assert env.eval_dataset is not None
    assert len(env.dataset) == 10
    assert len(env.eval_dataset) == 3
    assert "question" in env.dataset.column_names


def test_dataset_generation_completion_mode(toy_env_class):
    """gym_to_hf generates datasets with prompt column for completion mode."""
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="completion",
        num_train_episodes=5,
        num_eval_episodes=2,
    )

    assert env.dataset is not None
    assert len(env.dataset) == 5
    assert "prompt" in env.dataset.column_names
    assert isinstance(env.dataset[0]["prompt"], str)


def test_train_and_eval_datasets_separate(toy_env_class):
    """Train and eval datasets are separate objects with correct sizes."""
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=11,
        num_eval_episodes=3,
    )

    assert env.dataset is not env.eval_dataset
    assert len(env.dataset) == 11
    assert len(env.eval_dataset) == 3
