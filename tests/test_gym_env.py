from __future__ import annotations
import re
from typing import Any, Dict, List

import pytest
from datasets import Dataset
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion import Completion as OAICompletion
from openai.types.completion_choice import CompletionChoice
from verifiers.envs.gym_env import GymEnv, EpisodicSumRubric


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

    def reset(self, **kwargs):
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
        return obs, reward, done, False, info


# ----------------- Mock OpenAI client -----------------
class _MockChatCompletions:
    async def create(self, *, model: str, messages: List[Dict[str, str]], **kwargs):
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

        # Policy: if x<3, action 1. if x>=3, action 0
        action = "0"
        if "x=" in str(last_user) and n < 3:
            action = "1"

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
def eval_dataset():
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
    assert len(steps) > 0

    # Interaction Trace:
    # 1. x=0 -> Act 1 -> x=1 (Reward 0)
    # 2. x=1 -> Act 1 -> x=2 (Reward 0)
    # 3. x=2 -> Act 1 -> x=3 (Reward 1.0, Done=True)
    # 4. Observability Step: MultiTurnEnv intentionally generates one final response.
    #    Since done=True, env_response should append the sentinel.
    assert len(steps) == 4
    
    # Check that the final prompt contains the sentinel message
    last_prompt = steps[-1]["prompt"]
    assert "Episode already ended." in str(last_prompt)
    
    # Reward sum check
    assert res["reward"] == [1.0]
    assert st.get("gym_done", False) is True


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

    # Interaction Trace:
    # 1. Model generates "1".
    # 2. Env attempts parse -> Fails -> Returns Error Msg -> Sets Done=True.
    # 3. Observability Step: MultiTurnEnv generates one final response to the Error Msg.
    assert len(steps) == 2
    
    assert st.get("gym_done", False) is True
    
    # The error message is passed as feedback to the final observability turn.
    last_prompt = steps[-1]["prompt"]
    assert "Action Parsing Error" in str(last_prompt)
    # Note: In the current implementation, 'except Exception' returns early, 
    # so "Episode already ended." is NOT appended after the error message.
    # We verify that we got the specific error message instead.


def test_mode_1_respects_max_episode_steps(eval_dataset, client):
    """The loop should respect max_episode_steps even if env never terminates."""
    class NoTermEnv:
        def reset(self, **kwargs):
            return "x=0", {}
        def step(self, action: int):
            return "x=1", 0.0, False, False, {}

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
    
    # max_turns=3 serves as a hard cutoff.
    assert len(steps) == 3
    assert st.get("gym_done", False) is False
    assert st.get("is_completed", False) is True
    
    # CRITICAL CHECK: Since the env didn't finish itself (gym_done=False),
    # the sentinel message should NOT be present in the last prompt.
    last_prompt = steps[-1]["prompt"]
    assert "Episode already ended." not in str(last_prompt)


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
    first_prompt = st["prompt"]
    
    roles = [m["role"] for m in first_prompt]
    contents = [m.get("content") for m in first_prompt]
    assert roles[:4] == ["system", "user", "assistant", "user"]
    assert contents[0] == "SYS"
    assert contents[-1].startswith("x=0")


def test_mode_1_five_tuple_step_normalization(eval_dataset, client):
    class FourTupleEnv:
        def reset(self, **kwargs):
            return "x=0", {}
        def step(self, action: int):
            return "x=1", 1.0, True, {"info": "done"}

    env = GymEnv(
        env_cls=FourTupleEnv,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    steps = st.get("trajectory", [])
    
    # Interaction Trace:
    # 1. Action -> Env Step (Done=True).
    # 2. Observability Step.
    assert len(steps) == 2
    
    # Check that the final prompt contains the sentinel message
    last_prompt = steps[-1]["prompt"]
    assert "Episode already ended." in str(last_prompt)
    
    # The info should be attached to the step that generated it (Step 0)
    assert steps[0]["extras"]["gym_info"] == {"info": "done"}
    assert st.get("gym_done", False) is True


def test_mode_1_homogeneous_with_dataset_init(toy_env_class, client):
    ds = Dataset.from_dict(
        {
            "prompt": [[]],
            "task": ["toy"],
            "info": [{"start": 5}],
            "answer": [""],
            "example_id": [0],
        }
    )

    env = GymEnv(
        env_cls=toy_env_class,
        env_kwargs={"target": 10},
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=ds,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    
    first_obs_msg = st["prompt"][-1]["content"]
    assert first_obs_msg == "x=5"
    assert st["info"]["start"] == 5


def test_mode_3_custom_subclass_obs_formatter(eval_dataset, client):
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
        eval_dataset=eval_dataset,
    )
    res = env.evaluate_sync(client=client, model="mock")
    st = res["state"][0]
    assert st["prompt"][-1]["content"] == "obs_is_0"


def test_error_no_env_cls(eval_dataset):
    with pytest.raises(ValueError):
        GymEnv(action_parser=parse_action, eval_dataset=eval_dataset)


def test_completion_mode_rollout_and_completion_text(toy_env_class, client):
    ds = Dataset.from_dict(
        {
            "prompt": [""],
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

    assert isinstance(st["prompt"], str)
    assert st["prompt"] == "x=0"
    comp = st.get("completion", "")
    assert isinstance(comp, str)


def test_auto_dataset_created_when_no_datasets(toy_env_class):
    num_train = 37
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        num_train_episodes=num_train,
    )

    assert env.dataset is not None
    assert env.eval_dataset is not None
    train_ds = env.dataset
    eval_ds = env.eval_dataset
    assert len(train_ds) == num_train
    assert len(eval_ds) == 1
    assert isinstance(train_ds[0]["prompt"], list)
    assert train_ds[0]["prompt"] == []


def test_auto_dataset_completion_prompt_shape(toy_env_class):
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="completion",
        num_train_episodes=5,
    )
    assert env.dataset is not None
    ds = env.dataset
    assert len(ds) == 5
    assert isinstance(ds[0]["prompt"], str)
    assert ds[0]["prompt"] == ""


def test_eval_dataset_does_not_become_train_data(toy_env_class, eval_dataset):
    num_train = 11
    env = GymEnv(
        env_cls=toy_env_class,
        action_parser=parse_action,
        message_type="chat",
        eval_dataset=eval_dataset,
        num_train_episodes=num_train,
    )
    assert env.eval_dataset is not None
    assert env.dataset is not None
    assert len(env.eval_dataset) == 1
    assert len(env.dataset) == num_train
    assert env.dataset is not env.eval_dataset