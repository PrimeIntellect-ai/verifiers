from __future__ import annotations
import re
from typing import Any, Dict, List

import pytest
from datasets import Dataset
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion import Completion as OAICompletion
from openai.types.completion_choice import CompletionChoice
from verifiers.envs.gym_env import GymEnv, EpisodicSumRubric, StepResetEnv


# ----------------- Toy Environment -----------------
class ToyEnv:
    """
    Simple counter: observation is text "x=<int>".
    Action is an integer delta (we'll use 0/1). Episode ends when x >= target or max_steps hit.
    Reward is 1.0 on the step that first reaches/exceeds target, else 0.0.
    """

    def __init__(self, start: int = 0, target: int = 3, max_steps: int = 20, **kwargs):
        self.start = int(start)
        self.target = int(target)
        self.max_steps = int(max_steps)
        self.x = self.start
        self.steps = 0
        self.done = False

    # Widen signature to match protocol (accept **kwargs)
    def reset(self, **kwargs):
        # Allow 'start' to be overridden by kwargs from dataset info
        self.start = int(kwargs.get("start", self.start))
        self.x = self.start
        self.steps = 0
        self.done = False
        obs = f"x={self.x}"
        info = {"target": self.target, "start": self.start}
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


# Another env for registry test
class AnotherToyEnv:
    def __init__(self, prefix: str = "y", **kwargs):
        self.prefix = prefix
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return f"{self.prefix}={self.steps}", {}

    def step(self, action: int):
        self.steps += 1
        done = self.steps >= 2
        return f"{self.prefix}={self.steps}", 0.0, done, {}


# ----------------- Mock OpenAI client (chat + completion) -----------------
class _MockChatCompletions:
    async def create(self, *, model: str, messages: List[Dict[str, str]], **kwargs):
        # Read latest user obs "x=<n>" or "y=<n>"
        last_user = next(
            (
                m.get("content") or ""
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        n = 0
        m = re.search(r"[xy]\s*=\s*(-?\d+)", last_user)
        if m:
            n = int(m.group(1))

        # Policy: if x<3, action 1. if y or x>=3, action 0
        action = "0"
        if "x=" in last_user and n < 3:
            action = "1"

        message = ChatCompletionMessage.model_construct(
            role="assistant",
            content=action,
            tool_calls=None,
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


class _MockCompletions:
    async def create(self, *, model: str, prompt: str, **kwargs):
        n = 0
        m = re.search(r"x\s*=\s*(-?\d+)", prompt or "")
        if m:
            n = int(m.group(1))
        action = "1" if n < 3 else "0"

        choice = CompletionChoice.model_construct(
            index=0,
            text=action,
            logprobs=None,
            finish_reason="stop",
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


# ----------------- Helpers -----------------
def parse_action(txt: str) -> int:
    """Pick first integer; clamp to {0,1}."""
    m = re.search(r"[-+]?\d+", txt)
    if not m:
        raise ValueError(f"No int in: {txt!r}")
    return 1 if int(m.group(0)) > 0 else 0


# ----------------- Fixtures -----------------
@pytest.fixture
def toy_env_class():
    return ToyEnv


@pytest.fixture
def eval_dataset():
    # A generic 1-row dataset for simple tests
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


# ----------------- Tests (chat mode) -----------------
def test_mode_1_basic_rollout_and_reward_sum(toy_env_class, eval_dataset, client):
    """Tests Mode 1 (Homogeneous) with default env_kwargs"""
    env = GymEnv(
        env_cls=toy_env_class,
        env_kwargs={"start": 0, "target": 3, "max_steps": 10},
        action_parser=parse_action,
        message_type="chat",
        system_prompt="Reply ONLY with 0 or 1.",
        eval_dataset=eval_dataset,
        max_episode_steps=10,
        rubric=EpisodicSumRubric(),
    )

    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("trajectory", [])
    assert len(steps) > 0  # Ensure rollout happened

    # Return equals sum of per-step rewards (rubric responsibility).
    logged = sum(s.get("reward", 0.0) for s in steps)
    assert res["reward"] == [pytest.approx(logged)]

    # Sanity: step records are well-formed and last step has boolean flags.
    assert all(isinstance(s.get("reward", 0.0), (int, float)) for s in steps)
    last = steps[-1]
    assert isinstance(last["extras"].get("terminated", False), bool)
    assert isinstance(last["extras"].get("truncated", False), bool)


def test_invalid_parse_truncates(toy_env_class, eval_dataset, client):
    def bad_parser(_txt: str) -> int:
        raise ValueError("no action")

    env = GymEnv(
        env_cls=toy_env_class,
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
    steps = st.get("trajectory", [])

    # The loop runs once, adds a trajectory step, then fails on parse.
    assert len(steps) == 1
    # The rubric will sum rewards; the only step has reward=None, so sum is 0.
    assert res["reward"] == [0.0]
    # The error must be logged.
    assert "errors" in st
    assert any("action_parse_error" in err for err in st["errors"])


def test_mode_1_env_setup_called_once_per_rollout(eval_dataset, client):
    calls = {"n": 0}

    class EnvWithSetup(ToyEnv):
        def setup(self):
            calls["n"] += 1

    env = GymEnv(
        env_cls=EnvWithSetup,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )

    # One evaluate() call => one rollout => one env instance => setup() once
    env.evaluate_sync(client=client, model="mock")
    assert calls["n"] == 1


def test_mode_1_respects_max_episode_steps(eval_dataset, client):
    """The loop should respect max_episode_steps even if env never terminates."""

    class NoTermEnv:
        def reset(self, **kwargs):
            return "x=0", {}

        def step(self, action: int):
            return "x=1", 0.0, False, {}

    env = GymEnv(
        env_cls=NoTermEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        max_episode_steps=3,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("trajectory", [])
    assert len(steps) == 3
    assert isinstance(steps[-1]["extras"].get("terminated", False), bool)
    assert isinstance(steps[-1]["extras"].get("truncated", False), bool)


def test_mode_1_history_contains_system_fewshot_and_obs(
    toy_env_class, eval_dataset, client
):
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
    assert contents[-1].startswith("x=0")  # Default start


def test_mode_1_five_tuple_step_normalization(eval_dataset, client):
    class FiveTupleEnv:
        def reset(self, **kwargs):
            return "x=0", {}

        def step(self, action: int):
            return "x=1", 1.0, True, False, {"info": "done"}

    env = GymEnv(
        env_cls=FiveTupleEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("trajectory", [])
    assert len(steps) == 1
    assert steps[0]["extras"].get("terminated", False) is True
    assert steps[0]["extras"].get("truncated", False) is False
    assert steps[0]["extras"].get("step_info") == {"info": "done"}


def test_mode_1_env_setup_fn_precedence(eval_dataset, client):
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


# ----------------- Tests for Environment Creation Modes -----------------


def test_mode_1_homogeneous_with_dataset_init(toy_env_class, client):
    """Tests Mode 1 (Homogeneous) + dataset `info` passing to reset()"""
    # Dataset `info` specifies start=5
    ds = Dataset.from_dict(
        {
            "prompt": [[]],
            "task": ["toy"],
            "info": [{"start": 5}],  # <-- Kwarg for reset()
            "answer": [""],
            "example_id": [0],
        }
    )

    env = GymEnv(
        env_cls=toy_env_class,
        env_kwargs={"target": 10},  # Default target
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=ds,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    traj = st["trajectory"]
    assert len(traj) > 0
    # First prompt should contain the *initial* observation from reset()
    first_obs_msg = traj[0]["prompt"][-1]["content"]
    assert first_obs_msg == "x=5"
    # Check that info from reset was merged
    assert st["info"]["start"] == 5
    assert st["info"]["reset_info"]["target"] == 10


def test_mode_1_homogeneous_multi_row_init(toy_env_class, client):
    """Tests Mode 1 with 5 rollouts, ensuring each uses its correct dataset row for init"""
    N = 5
    ds = Dataset.from_dict(
        {
            "prompt": [[]] * N,
            "task": ["toy"] * N,
            "info": [{"start": i} for i in range(N)],  # <-- [0, 1, 2, 3, 4]
            "answer": [""] * N,
            "example_id": list(range(N)),
        }
    )

    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=ds,
    )

    # num_examples=-1 means use the whole dataset
    res = env.evaluate_sync(client=client, model="mock", num_examples=-1)

    # We should get N states back, one for each example_id
    assert len(res["state"]) == N

    # Sort states by example_id to ensure order
    all_states = sorted(res["state"], key=lambda s: s["example_id"])

    expected_starts = list(range(N))
    for i, state in enumerate(all_states):
        expected_start = expected_starts[i]
        
        # Check that the correct input info was passed
        assert state["input"]["info"]["start"] == expected_start
        assert state["example_id"] == i

        # Check that the env's reset() method received it
        assert state["info"]["reset_info"]["start"] == expected_start

        # Check that the first observation matches
        traj = state["trajectory"]
        assert len(traj) > 0
        first_obs_msg = traj[0]["prompt"][-1]["content"]
        assert first_obs_msg == f"x={expected_start}"


def test_mode_2_heterogeneous_registry(client):
    """Tests Mode 2 (Heterogeneous) using env_registry and dataset `info`"""
    REGISTRY = {"toy_v1": ToyEnv, "toy_v2": AnotherToyEnv}

    # Dataset specifies which env to run and its __init__ kwargs
    ds = Dataset.from_dict(
        {
            "prompt": [[], []],
            "task": ["multi", "multi"],
            "info": [
                {"env_type": "toy_v1", "env_kwargs": {"start": 1, "target": 2}},
                {"env_type": "toy_v2", "env_kwargs": {"prefix": "z"}},
            ],
            "answer": ["", ""],
            "example_id": [0, 1],
        }
    )

    env = GymEnv(
        env_registry=REGISTRY,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=ds,
    )
    res = env.evaluate_sync(client=client, model="mock", num_examples=2)
    assert len(res["state"]) == 2

    # Sort states just in case
    all_states = sorted(res["state"], key=lambda s: s["example_id"])

    # Check state 0 (ToyEnv)
    st0 = all_states[0]
    traj0 = st0["trajectory"]
    assert traj0[0]["prompt"][-1]["content"] == "x=1"  # From env_kwargs.start
    assert st0["info"]["reset_info"]["target"] == 2  # From env_kwargs.target

    # Check state 1 (AnotherToyEnv)
    st1 = all_states[1]
    traj1 = st1["trajectory"]
    assert traj1[0]["prompt"][-1]["content"] == "z=0"  # From env_kwargs.prefix
    assert traj1[1]["prompt"][-1]["content"] == "z=1"


def test_mode_3_custom_subclass_make_env(eval_dataset, client):
    """Tests Mode 3 (Custom) by subclassing and overriding _make_env"""
    calls = {"n": 0}

    class CustomGymEnv(GymEnv):
        def _make_env(self, state: State | None = None) -> StepResetEnv:
            calls["n"] += 1
            # Custom logic: always start at 100, regardless of dataset
            return ToyEnv(start=100)

    # Note: No env_cls or env_registry provided
    env = CustomGymEnv(
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")

    assert calls["n"] == 1  # Custom _make_env was called
    st = res["state"][0]
    traj = st["trajectory"]
    assert traj[0]["prompt"][-1]["content"] == "x=100"  # Custom logic applied


def test_mode_3_custom_subclass_obs_formatter(eval_dataset, client):
    """Tests Mode 3 (Custom) by subclassing to override obs_to_text"""

    class NumObsEnv:
        def reset(self, **kwargs):
            return 0, {}

        def step(self, action: int):
            return 1, 0.0, True, {}

    # Subclass to override obs_to_text, but use Mode 1 for _make_env
    class FmtGymEnv(GymEnv):
        def obs_to_text(self, obs: Any) -> str:
            # Custom formatter
            return f"obs_is_{obs}"

    env = FmtGymEnv(
        env_cls=NumObsEnv,  # Use Mode 1 for creation
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    traj = st["trajectory"]
    
    assert len(traj) == 1
    assert traj[0]["prompt"][-1]["content"] == "obs_is_0"


def test_error_no_mode_selected(eval_dataset):
    """Tests error if no env_cls, env_registry, or subclass override is given"""
    with pytest.raises(NotImplementedError):
        # No env_cls, no env_registry, and not subclassing _make_env
        env = GymEnv(action_parser=parse_action, eval_dataset=eval_dataset)
        # We need to call _make_env to trigger the error, so we run a rollout
        # Easiest way is to just call the method directly for this test
        env._make_env(state=None)


def test_error_both_modes_selected():
    """Tests error if both env_cls and env_registry are given"""
    with pytest.raises(ValueError):
        GymEnv(
            env_cls=ToyEnv,
            env_registry={"toy": ToyEnv},
            action_parser=parse_action,
        )


# ----------------- Completion mode test -----------------


def test_completion_mode_rollout_and_completion_text(toy_env_class, client):
    """
    Smoke test for message_type='completion' (Mode 1):
    - Uses text completions instead of chat.
    - Ensures state['completion'] is a concatenated string.
    - Ensures trajectory prompt/completion fields are strings.
    """
    ds = Dataset.from_dict(
        {
            "prompt": [""],  # completion-style prompt
            "task": ["toy"],
            "info": [{}],
            "answer": [""],
            "example_id": [0],
        }
    )

    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="completion",
        eval_dataset=ds,
    )

    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]

    # In completion mode, completion is a single concatenated string
    comp = st["completion"]
    assert isinstance(comp, str)
    assert comp != ""
    # For ToyEnv(start=0,target=3) with our policy, this will be "111"
    assert set(comp).issubset({"0", "1"})

    traj = st["trajectory"]
    assert len(traj) >= 1
    first_step = traj[0]
    assert isinstance(first_step["prompt"], str)
    assert isinstance(first_step["completion"], str)


# ----------------- Evaluation customization tests -----------------


def test_eval_runner_sync_delegates(toy_env_class, eval_dataset, client):
    """When eval_runner is provided (sync), GymEnv should return its result verbatim."""
    sentinel = object()
    called = {"n": 0, "args": None}

    def custom_eval_runner(self_env: GymEnv, **kwargs):
        called["n"] += 1
        called["args"] = kwargs
        return sentinel

    env = GymEnv(
        env_cls=toy_env_class,
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
async def test_eval_runner_async_delegates(toy_env_class, eval_dataset, client):
    """When eval_runner is async, GymEnv should await and return its result."""

    class Sentinel:
        pass

    sentinel = Sentinel()
    called = {"n": 0}

    async def custom_eval_runner(self_env: GymEnv, **kwargs):
        called["n"] += 1
        return sentinel

    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        eval_runner=custom_eval_runner,
    )
    out = await env.evaluate(client=client, model="mock", num_examples=3)
    assert out is sentinel
    assert called["n"] == 1


def test_dummy_eval_num_examples_maps_to_rollouts(toy_env_class, client):
    """
    With the built-in dummy eval dataset (auto_dummy_eval), num_examples=N maps to
    rollouts_per_example=N (and num_examples becomes 1).
    """
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        # No explicit eval_dataset: use _default_eval_ds()
    )
    res = env.evaluate_sync(client=client, model="mock", num_examples=5)
    # Should have run 5 rollouts of the single dummy example
    assert len(res["state"]) == 5
    assert res["metadata"]["num_examples"] == 1
    assert res["metadata"]["rollouts_per_example"] == 5


def test_dummy_eval_explicit_rollouts_wins(toy_env_class, client):
    """
    In dummy mode, if caller explicitly sets rollouts_per_example, we still
    collapse num_examples to 1 but keep the caller's RPE.
    """
    env = GymEnv(
        env_cls=toy_env_class,
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


def test_non_dummy_eval_no_mapping(toy_env_class, client):
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
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=ds,
    )
    # Ask for more examples than exist; Environment caps to len(dataset)=2
    res = env.evaluate_sync(client=client, model="mock", num_examples=5)
    assert len(res["state"]) == 2
    assert res["metadata"]["num_examples"] == 2
    assert res["metadata"]["rollouts_per_example"] == 1

# ----------------- Auto-dataset and dataset semantics tests -----------------


def test_auto_dataset_created_when_no_datasets(toy_env_class):
    """
    If neither dataset nor eval_dataset is provided, GymEnv should:
      - auto-build a train dataset with num_train_episodes rows
      - create a 1-row dummy eval_dataset
      - still behave as a valid Environment (dataset non-None)
    """
    num_train = 37
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=num_train,
        # rely on auto_dummy_eval=True (default)
    )

    # Environment must have both dataset and eval_dataset
    assert env.dataset is not None
    assert env.eval_dataset is not None

    # Train dataset should have num_train rows, eval should be the 1-row dummy
    train_ds = env.dataset
    eval_ds = env.eval_dataset
    assert len(train_ds) == num_train
    assert len(eval_ds) == 1

    # Columns should match what _build_auto_dataset/_default_eval_ds produce
    assert set(train_ds.column_names) == {"prompt", "answer", "info", "task", "example_id"}
    assert set(eval_ds.column_names) == {"prompt", "answer", "info", "task", "example_id"}

    # Chat mode: prompts are lists (here empty lists)
    assert isinstance(train_ds[0]["prompt"], list)
    assert train_ds[0]["prompt"] == []

    # Info should be a dict per row
    assert isinstance(train_ds[0]["info"], dict)


def test_auto_dataset_completion_prompt_shape(toy_env_class):
    """
    In completion mode, auto-built train dataset should have string prompts (""), not lists.
    """
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="completion",
        num_train_episodes=5,
        # no datasets passed -> auto dataset + dummy eval
    )

    assert env.dataset is not None
    ds = env.dataset
    assert len(ds) == 5

    # Completion mode uses plain string prompts
    assert isinstance(ds[0]["prompt"], str)
    assert ds[0]["prompt"] == ""


def test_eval_dataset_is_mirrored_into_dataset(toy_env_class, eval_dataset):
    """
    If user passes only eval_dataset, GymEnv should mirror it into dataset so
    trainers that expect env.dataset (like Orchestrator) don't explode.
    """
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )

    # Both datasets should be present and have same length/contents
    assert env.eval_dataset is not None
    assert env.dataset is not None

    assert len(env.eval_dataset) == len(env.dataset)
    # Quick sanity check on first row fields
    assert env.dataset[0]["example_id"] == env.eval_dataset[0]["example_id"]
    assert env.dataset[0]["task"] == env.eval_dataset[0]["task"]


def test_user_dataset_prevents_auto_dataset(toy_env_class):
    """
    If user provides a dataset explicitly, GymEnv must use it and not override it
    with an auto-generated one.
    """
    ds = Dataset.from_dict(
        {
            "prompt": [[]],
            "task": ["toy"],
            "info": [{"start": 42}],
            "answer": [""],
            "example_id": [7],
        }
    )

    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        dataset=ds,
        # no eval_dataset -> Environment will allow eval_dataset=None
    )

    assert env.dataset is not None
    assert len(env.dataset) == 1
    row = env.dataset[0]
    assert row["example_id"] == 7
    assert row["info"]["start"] == 42


def test_auto_dataset_registry_and_info_builder(client):
    """
    In heterogeneous mode, _build_auto_dataset should:
      - create info.env_type cycling over registry keys
      - include info_builder(i) into info
    """

    class DummyEnvForRegistry:
        def __init__(self, tag: str = "z", **kwargs):
            self.tag = tag

        def reset(self, **kwargs):
            return f"{self.tag}=0", {}

        def step(self, action: int):
            return f"{self.tag}=1", 0.0, True, {}

    REGISTRY = {
        "env_a": ToyEnv,
        "env_b": DummyEnvForRegistry,
    }

    def info_builder(i: int) -> Dict[str, Any]:
        # Just mark the row so we can assert it's wired through
        return {"builder_idx": i}

    num_train = 6
    env = GymEnv(
        env_registry=REGISTRY,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=num_train,
        info_builder=info_builder,
        # no datasets -> auto dataset + dummy eval
    )

    assert env.dataset is not None
    ds = env.dataset
    assert len(ds) == num_train

    infos = ds["info"]
    assert len(infos) == num_train

    # env_type should cycle through registry keys in order
    keys = list(REGISTRY.keys())
    for i, info in enumerate(infos):
        assert info["env_type"] == keys[i % len(keys)]
        # info_builder contribution must be present and correct
        assert info["builder_idx"] == i

    # Sanity: running a tiny eval should still work with this auto dataset
    res = env.evaluate_sync(client=client, model="mock", num_examples=2)
    assert len(res["state"]) == 2
