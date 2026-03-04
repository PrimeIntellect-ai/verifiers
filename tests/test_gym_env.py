from __future__ import annotations

import re
from typing import Any

import pytest

from tests.conftest import MockClient
from verifiers.envs.experimental.gym_env import EpisodicSumRubric, GymEnv
from verifiers.types import Response, ResponseMessage


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


class GymMockClient(MockClient):
    """Mock client for GymEnv tests that responds based on x=N observations."""

    @staticmethod
    def _is_text_prompt(prompt) -> bool:
        """Detect completion-mode prompts (single TextMessage with role='text')."""
        if isinstance(prompt, str):
            return True
        if isinstance(prompt, list) and len(prompt) == 1:
            msg = prompt[0]
            role = msg["role"] if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "text":
                return True
        return False

    @staticmethod
    def _extract_text(prompt) -> str:
        """Extract text content from a text/completion prompt."""
        if isinstance(prompt, str):
            return prompt
        msg = prompt[0]
        if isinstance(msg, dict):
            return msg.get("content", "")
        return getattr(msg, "content", "")

    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        self.call_count += 1
        self.last_call_kwargs = {
            "prompt": prompt,
            "model": model,
            "sampling_args": sampling_args,
            "tools": tools,
            **kwargs,
        }

        # Extract last user content
        last_user = ""
        if self._is_text_prompt(prompt):
            last_user = self._extract_text(prompt)
        elif isinstance(prompt, list):
            for msg in reversed(prompt):
                role = (
                    msg.get("role")
                    if isinstance(msg, dict)
                    else getattr(msg, "role", "")
                )
                if role == "user":
                    content = (
                        msg.get("content")
                        if isinstance(msg, dict)
                        else getattr(msg, "content", "")
                    )
                    last_user = str(content)
                    break

        n = 0
        m = re.search(r"x\s*=\s*(-?\d+)", last_user)
        if m:
            n = int(m.group(1))
        action = "1" if n < 3 else "0"

        return Response(
            id="mock-id",
            created=0,
            model=model,
            usage=None,
            message=ResponseMessage(
                content=action,
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )


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
    return GymMockClient()


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

    outputs = env.evaluate_sync(
        client=client, model="mock", state_columns=["trajectory", "gym_done"]
    )
    st = outputs["outputs"][0]
    steps = st.get("trajectory", [])

    assert len(steps) > 0
    assert st["reward"] == 1.0
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

    res = env.evaluate_sync(
        client=client, model="mock", state_columns=["trajectory", "gym_done"]
    )
    st = res["outputs"][0]
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
    res = env.evaluate_sync(
        client=client, model="mock", state_columns=["trajectory", "gym_done"]
    )
    st = res["outputs"][0]
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
    st = res["outputs"][0]
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
    res = env.evaluate_sync(
        client=client, model="mock", state_columns=["trajectory", "gym_done"]
    )
    st = res["outputs"][0]
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
    st = res["outputs"][0]

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
    st = res["outputs"][0]
    assert st["prompt"][-1]["content"] == "obs_is_0"


def test_missing_env_cls_raises_error():
    """GymEnv requires env_cls argument."""
    with pytest.raises(TypeError):
        GymEnv(action_parser=parse_action)  # type: ignore[call-arg]


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
