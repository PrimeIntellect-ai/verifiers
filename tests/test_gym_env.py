from __future__ import annotations
import re
from typing import Any, Dict, List

import pytest
from datasets import Dataset
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from verifiers.envs.gym_env import GymEnv, EpisodicSumRubric


# ----------------- Toy Environment -----------------
class ToyEnv:
    """
    Simple counter: observation is text "x=<int>".
    Action is an integer delta (we'll use 0/1). Episode ends when x >= target or max_steps hit.
    Reward is 1.0 on the step that first reaches/exceeds target, else 0.0.
    """

    def __init__(self, start: int = 0, target: int = 3, max_steps: int = 20):
        self.start = int(start)
        self.target = int(target)
        self.max_steps = int(max_steps)
        self.x = self.start
        self.steps = 0
        self.done = False

    # Widen signature to match protocol (accept **kwargs)
    def reset(self, **kwargs):
        self.x = self.start
        self.steps = 0
        self.done = False
        obs = f"x={self.x}"
        info = {"target": self.target}
        return obs, info

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode finished, call reset().")
        self.steps += 1
        self.x += int(action)
        done = self.x >= self.target or self.steps >= self.max_steps
        self.done = done
        reward = 1.0 if self.x >= self.target else 0.0
        info = {"x": self.x, "target": self.target, "reached": self.x >= self.target}
        obs = f"x={self.x}"
        return obs, reward, done, info  # 4-tuple; wrapper normalizes it


# ----------------- Mock OpenAI client (chat) -----------------
class _MockChatCompletions:
    async def create(self, *, model: str, messages: List[Dict[str, str]], **kwargs):
        # Read latest user obs "x=<n>" and return a valid action as content.
        last_user = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        n = 0
        m = re.search(r"x\s*=\s*(-?\d+)", last_user)
        if m:
            n = int(m.group(1))
        action = "1" if n < 3 else "0"

        # Real ChatCompletion + Choice + ChatCompletionMessage, built minimally.
        message = ChatCompletionMessage.model_construct(
            role="assistant",
            content=action,
            tool_calls=None,  # must exist so response.choices[0].message.tool_calls is safe
        )
        choice = Choice.model_construct(
            index=0,
            message=message,
            finish_reason="stop",
            logprobs=None,
        )
        return ChatCompletion.model_construct(
            id="mock-chatcmpl",
            choices=[choice],
            created=0,
            model=model,
            object="chat.completion",
            usage=None,
        )


class _MockChat:
    def __init__(self):
        self.completions = _MockChatCompletions()


class MockAsyncOpenAI:
    def __init__(self):
        self.chat = _MockChat()
        self.base_url = "mock://local"


# ----------------- Helpers -----------------
def parse_action(txt: str) -> int:
    """Pick first integer; clamp to {0,1}."""
    m = re.search(r"[-+]?\d+", txt)
    if not m:
        raise ValueError(f"No int in: {txt!r}")
    return 1 if int(m.group(0)) > 0 else 0


# ----------------- Fixtures -----------------
@pytest.fixture
def toy_env():
    return ToyEnv(start=0, target=3, max_steps=10)


@pytest.fixture
def eval_dataset():
    # Environment base class demands dataset or eval_dataset
    # This is *not* the built-in dummy dataset used by GymBaseEnv.
    return Dataset.from_dict(
        {
            "prompt": [[]],
            "task": ["toy"],
            "info": [{}],
            "answer": [""],
            "example_id": [0],
        }
    )


@pytest.fixture
def client():
    return MockAsyncOpenAI()


# ----------------- Tests -----------------
def test_basic_rollout_and_reward_sum(toy_env, eval_dataset, client):
    env = GymEnv(
        env_cls=ToyEnv,
        env_kwargs={
            "start": toy_env.start,
            "target": toy_env.target,
            "max_steps": toy_env.max_steps,
        },
        action_parser=parse_action,
        message_type="chat",
        system_prompt="Reply ONLY with 0 or 1.",
        eval_dataset=eval_dataset,
        max_episode_steps=toy_env.max_steps,
        rubric=EpisodicSumRubric(),  # default rubric sums stepwise rewards
    )

    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("responses", [])

    # Return equals sum of per-step rewards (rubric responsibility).
    logged = sum(s.get("reward", 0.0) for s in steps)
    assert res["reward"] == [pytest.approx(logged)]

    # Sanity: step records are well-formed and last step has boolean flags.
    assert all(isinstance(s.get("reward", 0.0), (int, float)) for s in steps)
    if steps:
        last = steps[-1]
        assert isinstance(last.get("terminated", False), bool)
        assert isinstance(last.get("truncated", False), bool)


def test_invalid_parse_truncates(eval_dataset, client):
    def bad_parser(_txt: str) -> int:
        raise ValueError("no action")

    env = GymEnv(
        env_cls=ToyEnv,
        action_parser=bad_parser,
        message_type="chat",
        eval_dataset=eval_dataset,
    )

    res = env.evaluate_sync(
        client=client,
        model="mock-gpt",
        num_examples=1,
        rollouts_per_example=1,
    )
    st = res["state"][0]
    steps = st.get("responses", [])

    # If a parse error happens immediately, there may be zero steps.
    # Either way, reward should be zero and an error should be logged.
    assert res["reward"] == [0.0]
    assert any("action_parse_error" in err for err in st.get("errors", []))

    # If any step was logged before truncation, last step must show truncated flag.
    if steps:
        assert steps[-1].get("truncated", False) is True


def test_env_setup_called_once_per_rollout(eval_dataset, client):
    calls = {"n": 0}

    class EnvWithSetup(ToyEnv):
        def setup(self):
            calls["n"] += 1

        def reset(self, **kwargs):
            return super().reset(**kwargs)

    env = GymEnv(
        env_cls=EnvWithSetup,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )

    # One evaluate() call => one rollout => one env instance => setup() once
    env.evaluate_sync(client=client, model="mock")
    assert calls["n"] == 1


def test_nonstring_obs_requires_formatter(eval_dataset, client):
    class NumObsEnv:
        def __init__(self):
            self.x = 0
            self.done = False

        def reset(self, **kwargs):
            self.x = 0
            self.done = False
            return 0, {}

        def step(self, action: int):
            self.x += action
            d = self.x >= 1
            self.done = d
            return self.x, float(d), d, {}

    # Without custom obs_to_text -> TypeError during rollout when obs_to_text is called
    env = GymEnv(
        env_cls=NumObsEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    with pytest.raises(TypeError):
        env.evaluate_sync(client=client, model="mock")

    # With custom obs_to_text -> OK; we inspect the first trajectory prompt
    class FmtGymEnv(GymEnv):
        def obs_to_text(self, obs: Any) -> str:
            return f"x={obs}"

    env = FmtGymEnv(
        env_cls=NumObsEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    traj = st["trajectory"]
    assert len(traj) >= 1
    first_prompt = traj[0]["prompt"]
    assert isinstance(first_prompt, list)
    # last message in initial prompt is the user obs
    assert first_prompt[-1]["role"] == "user"
    assert first_prompt[-1].get("content") == "x=0"


def test_respects_max_episode_steps(eval_dataset, client):
    """The loop should respect max_episode_steps even if env never terminates."""

    class NoTermEnv:
        def __init__(self):
            self.x = 0
            self.steps = 0
            self.done = False

        def reset(self, **kwargs):
            self.x = 0
            self.steps = 0
            self.done = False
            return f"x={self.x}", {}

        def step(self, action: int):
            self.steps += 1
            self.x += int(action)
            # never terminates; returns (obs, reward, done, info)
            return f"x={self.x}", 0.0, False, {"x": self.x}

    env = GymEnv(
        env_cls=NoTermEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        max_episode_steps=3,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("responses", [])
    assert len(steps) == 3
    # Flags are present and boolean
    assert isinstance(steps[-1].get("terminated", False), bool)
    assert isinstance(steps[-1].get("truncated", False), bool)


def test_history_contains_system_fewshot_and_obs(eval_dataset, client):
    few = [
        {"role": "user", "content": "demo Q"},
        {"role": "assistant", "content": "demo A"},
    ]

    env = GymEnv(
        env_cls=ToyEnv,
        action_parser=parse_action,
        message_type="chat",
        system_prompt="SYS",
        few_shot=few,
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    traj = st["trajectory"]
    assert len(traj) >= 1
    first_prompt = traj[0]["prompt"]
    roles = [m["role"] for m in first_prompt]
    contents = [m.get("content") for m in first_prompt]
    # system, few-shot user/assistant, then user with obs
    assert roles[:4] == ["system", "user", "assistant", "user"]
    assert contents[0] == "SYS"
    assert contents[-1].startswith("x=")


def test_five_tuple_step_normalization(eval_dataset, client):
    class FiveTupleEnv:
        def __init__(self):
            self.x = 0
            self.done = False

        def reset(self, **kwargs):
            self.x = 0
            self.done = False
            return "x=0", {}

        def step(self, action: int):
            self.x += action
            terminated = self.x >= 2
            truncated = False
            info = {"x": self.x}
            return f"x={self.x}", float(terminated), terminated, truncated, info

    env = GymEnv(
        env_cls=FiveTupleEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("responses", [])
    assert steps[-1].get("terminated", False) is True
    assert steps[-1].get("truncated", False) is False


def test_env_setup_fn_precedence(eval_dataset, client):
    calls = {"env_setup": 0, "hook": 0}

    class E(ToyEnv):
        def setup(self):
            calls["env_setup"] += 1

    def setup_hook(e: Any, state: Any):
        calls["hook"] += 1

    env = GymEnv(
        env_cls=E,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        env_setup_fn=setup_hook,
    )
    env.evaluate_sync(client=client, model="mock")
    # Only our hook runs (GymEnv prefers env_setup_fn over env.setup)
    assert calls["hook"] == 1
    assert calls["env_setup"] == 0


# ----------------- Evaluation customization tests -----------------


def test_eval_runner_sync_delegates(eval_dataset, client):
    """When eval_runner is provided (sync), GymEnv should return its result verbatim."""
    sentinel = object()
    called = {"n": 0, "args": None}

    def custom_eval_runner(self_env: GymEnv, **kwargs):
        called["n"] += 1
        called["args"] = kwargs
        return sentinel

    env = GymEnv(
        env_cls=ToyEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        eval_runner=custom_eval_runner,
    )
    out = env.evaluate_sync(client=client, model="mock", num_examples=7)
    assert out is sentinel
    assert called["n"] == 1
    assert isinstance(called["args"], dict)
    # ensure we passed through some key params
    assert called["args"]["num_examples"] == 7
    assert called["args"]["model"] == "mock"


@pytest.mark.asyncio
async def test_eval_runner_async_delegates(eval_dataset, client):
    """When eval_runner is async, GymEnv should await and return its result."""

    class Sentinel:
        pass

    sentinel = Sentinel()
    called = {"n": 0}

    async def custom_eval_runner(self_env: GymEnv, **kwargs):
        called["n"] += 1
        return sentinel

    env = GymEnv(
        env_cls=ToyEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        eval_runner=custom_eval_runner,
    )
    out = await env.evaluate(client=client, model="mock", num_examples=3)
    assert out is sentinel
    assert called["n"] == 1


def test_dummy_eval_num_examples_maps_to_rollouts(eval_dataset, client):
    """
    With the built-in dummy eval dataset (auto_dummy_eval), num_examples=N maps to
    rollouts_per_example=N (and num_examples becomes 1).
    """
    env = GymEnv(
        env_cls=ToyEnv,
        action_parser=parse_action,
        message_type="chat",
        # No explicit eval_dataset: use _default_eval_ds()
    )
    res = env.evaluate_sync(client=client, model="mock", num_examples=5)
    # Should have run 5 rollouts of the single dummy example
    assert len(res["state"]) == 5
    assert res["metadata"]["num_examples"] == 1
    assert res["metadata"]["rollouts_per_example"] == 5


def test_dummy_eval_explicit_rollouts_wins(eval_dataset, client):
    """
    In dummy mode, if caller explicitly sets rollouts_per_example, we still
    collapse num_examples to 1 but keep the caller's RPE.
    """
    env = GymEnv(
        env_cls=ToyEnv,
        action_parser=parse_action,
        message_type="chat",
        # Again, rely on auto_dummy_eval; no explicit eval_dataset.
    )
    res = env.evaluate_sync(
        client=client, model="mock", num_examples=10, rollouts_per_example=3
    )
    assert len(res["state"]) == 3
    assert res["metadata"]["num_examples"] == 1
    assert res["metadata"]["rollouts_per_example"] == 3


def test_non_dummy_eval_no_mapping(client):
    """
    When eval_dataset has more than one row, num_examples behaves normally; no remap.
    """
    # Build a 2-row eval dataset
    ds = Dataset.from_dict(
        {
            "prompt": [[], []],
            "task": ["toy", "toy"],
            "info": [{}, {}],
            "answer": ["", ""],
            "example_id": [0, 1],
        }
    )
    env = GymEnv(
        env_cls=ToyEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=ds,
    )
    # Ask for more examples than exist; Environment caps to len(dataset)=2
    res = env.evaluate_sync(client=client, model="mock", num_examples=5)
    assert len(res["state"]) == 2
    assert res["metadata"]["num_examples"] == 2
    assert res["metadata"]["rollouts_per_example"] == 1
