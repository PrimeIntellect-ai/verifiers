from __future__ import annotations
import re
from typing import Any, Dict, List

import pytest
from datasets import Dataset
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
class _MockMsg:
    def __init__(self, content: str):
        self.content = content


class _MockChoice:
    def __init__(self, content: str):
        self.message = _MockMsg(content)


class _MockResp:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


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
        return _MockResp(action)


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
        env=toy_env,
        action_parser=parse_action,
        message_type="chat",
        system_prompt="Reply ONLY with 0 or 1.",
        eval_dataset=eval_dataset,
        max_episode_steps=toy_env.max_steps,
        rubric=EpisodicSumRubric(),  # default rubric sums stepwise rewards
    )

    res = env.evaluate_sync(client=client, model="mock")
    st = res.state[0]
    steps = st.get("responses", [])

    # Return equals sum of per-step rewards (rubric responsibility).
    logged = sum(s.get("reward", 0.0) for s in steps)
    assert res.reward == [pytest.approx(logged)]

    # Sanity: step records are well-formed and last step has boolean flags.
    assert all(isinstance(s.get("reward", 0.0), (int, float)) for s in steps)
    if steps:
        last = steps[-1]
        assert isinstance(last.get("terminated", False), bool)
        assert isinstance(last.get("truncated", False), bool)


def test_reset_and_step_passthrough(toy_env, eval_dataset, client):
    env = GymEnv(
        env=toy_env,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    obs, info = env.reset()
    assert isinstance(obs, str) and obs.startswith("x=")
    assert "target" in info

    obs2, r, done, trunc, _ = env.step(1)
    assert obs2.startswith("x=")
    assert isinstance(r, float)
    assert done in (True, False)
    assert trunc in (True, False)


def test_invalid_parse_truncates(toy_env, eval_dataset, client):
    def bad_parser(_txt: str) -> int:
        raise ValueError("no action")

    env = GymEnv(
        env=toy_env,
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
    st = res.state[0]
    steps = st.get("responses", [])

    # If a parse error happens immediately, there may be zero steps.
    # Either way, reward should be zero and an error should be logged.
    assert res.reward == [0.0]
    assert any("action_parse_error" in err for err in st.get("errors", []))

    # If any step was logged before truncation, last step must show truncated flag.
    if steps:
        assert steps[-1].get("truncated", False) is True


def test_setup_fn_called_once(eval_dataset, client):
    calls = {"n": 0}

    class EnvWithSetup(ToyEnv):
        def setup(self):
            calls["n"] += 1

        def reset(self, **kwargs):
            return super().reset(**kwargs)

    base = EnvWithSetup()
    env = GymEnv(
        env=base,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )

    # First reset triggers setup
    env.reset()
    # Subsequent resets do not re-call setup
    env.reset()
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

    # Without custom obs_to_text -> TypeError on reset (guardrail)
    env = GymEnv(
        env=NumObsEnv(),
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    with pytest.raises(TypeError):
        env.reset()

    # With custom obs_to_text -> OK; returned obs is raw (0), formatted text is in chat history
    class FmtGymEnv(GymEnv):
        def obs_to_text(self, obs: Any) -> str:
            return f"x={obs}"

    env = FmtGymEnv(
        env=NumObsEnv(),
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    obs, _ = env.reset()
    assert obs == 0  # raw obs unchanged
    # Type-narrow history
    assert isinstance(env.history, list)
    msg0 = env.history[0]
    assert isinstance(msg0, dict)
    assert msg0["role"] == "user"
    assert msg0.get("content") == "x=0"  # formatted text used for chat


# ----------------- Extra coverage -----------------


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
        env=NoTermEnv(),
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        max_episode_steps=3,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res.state[0]
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
        env=ToyEnv(),
        action_parser=parse_action,
        message_type="chat",
        system_prompt="SYS",
        few_shot=few,
        eval_dataset=eval_dataset,
    )
    obs, _ = env.reset()
    assert isinstance(env.history, list)
    roles = [m["role"] for m in env.history]  # type: ignore[index]
    contents = [m.get("content") for m in env.history]  # type: ignore[union-attr]
    # system, few-shot user/assistant, then user with obs
    assert roles[:4] == ["system", "user", "assistant", "user"]
    assert contents[0] == "SYS"
    assert contents[-1] == f"x={obs.split('=')[1]}"

    # After a step, a new user message with updated obs should be appended
    env.step(1)
    assert env.history[-1]["role"] == "user"  # type: ignore[index]
    assert env.history[-1].get("content", "").startswith("x=")  # type: ignore[index]


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
        env=FiveTupleEnv(),
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res.state[0]
    steps = st.get("responses", [])
    assert steps[-1].get("terminated", False) is True
    assert steps[-1].get("truncated", False) is False


def test_setup_fn_precedence_and_kwargs(eval_dataset, client):
    calls = {"env_setup": 0, "hook": 0, "kw": None}

    class E(ToyEnv):
        def setup(self):
            calls["env_setup"] += 1

    def setup_hook(e: Any, **kw):
        calls["hook"] += 1
        calls["kw"] = kw

    base = E()
    env = GymEnv(
        env=base,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        setup_fn=setup_hook,
        setup_kwargs={"alpha": 7},
    )
    env.reset()
    env.reset()
    # Only our hook runs, once (GymEnv prefers setup_fn over env.setup)
    assert calls["hook"] == 1
    assert calls["env_setup"] == 0
    assert calls["kw"] == {"alpha": 7}


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
        env=ToyEnv(),
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
        env=ToyEnv(),
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
    With the built-in dummy-style eval dataset (len==1), num_examples=N maps to
    rollouts_per_example=N (and num_examples becomes 1).
    """
    env = GymEnv(
        env=ToyEnv(),
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,  # len==1
    )
    res = env.evaluate_sync(client=client, model="mock", num_examples=5)
    # Should have run 5 rollouts of the single example
    assert len(res.state) == 5
    assert res.metadata.num_examples == 1
    assert res.metadata.rollouts_per_example == 5


def test_dummy_eval_explicit_rollouts_wins(eval_dataset, client):
    """
    If caller explicitly sets rollouts_per_example, we don't remap even if num_examples>1.
    """
    env = GymEnv(
        env=ToyEnv(),
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,  # len==1
    )
    res = env.evaluate_sync(
        client=client, model="mock", num_examples=10, rollouts_per_example=3
    )
    assert len(res.state) == 3
    assert res.metadata.num_examples == 1
    assert res.metadata.rollouts_per_example == 3


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
        env=ToyEnv(),
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=ds,
    )
    # Ask for more examples than exist; Environment caps to len(dataset)=2
    res = env.evaluate_sync(client=client, model="mock", num_examples=5)
    assert len(res.state) == 2
    assert res.metadata.num_examples == 2
    assert res.metadata.rollouts_per_example == 1
