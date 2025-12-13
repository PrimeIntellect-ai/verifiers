from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import gymnasium as gym
from datasets import Dataset

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import MessageType, State

# ---------- Types & Protocols ----------
ResetOut = Union[Any, Tuple[Any, Dict[str, Any]]]
StepOut = Union[
    Tuple[Any, float, bool, bool, Dict[str, Any]],
    Tuple[Any, float, bool, Dict[str, Any]],
]

class StepResetEnv(Protocol):
    def reset(self, **kwargs) -> ResetOut: ...
    def step(self, action: Any) -> StepOut: ...

def _normalize_reset(out: ResetOut) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}

def _normalize_step(out: StepOut) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    if isinstance(out, (tuple, list)):
        if len(out) == 5:
            return out  # type: ignore
        elif len(out) == 4:
            obs, reward, done, info = out  # type: ignore
            return obs, float(reward), bool(done), False, info
    raise RuntimeError(f"env.step() returned {type(out)}, expected tuple of length 4 or 5")

# ---------- Default Components ----------
def sum_step_rewards(*, state: State, **_) -> float:
    return float(sum(float(step.get("reward", 0.0) or 0.0) for step in state.get("trajectory", [])))

class EpisodicSumRubric(Rubric):
    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(funcs=[sum_step_rewards], weights=[weight], **kwargs)

# ---------- GymEnv ----------
class GymEnv(vf.MultiTurnEnv):
    """
    Universal runner for Gym-compatible environments.
    """
    def __init__(
        self,
        # Factory Args
        gym_id: str | None = None,
        env_cls: type | None = None,
        env_kwargs: Dict[str, Any] | None = None,
        # Interface Args
        action_parser: Callable[[str], Any] = lambda x: x, # Default to identity
        obs_to_text: Callable[[Any], str] | None = None,
        # Dataset Args
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        auto_dummy_eval: bool = True,
        num_train_episodes: int = 1000,
        # Base Args
        rubric: Rubric | None = None,
        max_episode_steps: int | None = None,
        message_type: MessageType = "chat",
        **kwargs,
    ):
        if not (gym_id or env_cls):
            raise ValueError("Must provide either `gym_id` or `env_cls`")

        self.gym_id = gym_id
        self._env_cls = env_cls
        self._env_kwargs = dict(env_kwargs or {})
        
        self.action_parser = action_parser
        self._obs_to_text_fn = obs_to_text
        self.max_episode_steps_override = max_episode_steps

        # Auto-Dataset Generation
        if dataset is None and auto_dummy_eval:
            dataset = self._build_auto_dataset(num_train_episodes, message_type)
            if eval_dataset is None:
                eval_dataset = Dataset.from_dict({
                    "prompt": [""], "answer": [""], "info": [{}], 
                    "task": ["default"], "example_id": [0]
                })

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric or EpisodicSumRubric(),
            message_type=message_type,
            max_turns=max_episode_steps or 1000,
            **kwargs,
        )

    def _make_env(self, state: State) -> StepResetEnv:
        if self.gym_id:
            return gym.make(self.gym_id, **self._env_kwargs)
        if self._env_cls:
            return self._env_cls(**self._env_kwargs) # type: ignore
        raise ValueError("Configuration error")

    async def setup_state(self, state: State) -> State:
        env = self._make_env(state)
        state["gym_env"] = env
        
        input_info = state.get("info", {}) or {}
        obs, _ = _normalize_reset(env.reset(**input_info))
        
        obs_text = self.obs_to_text(obs)
        if self.message_type == "chat":
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            if self.few_shot:
                msgs.extend(self.few_shot)
            msgs.append({"role": "user", "content": obs_text})
            state["prompt"] = msgs
        else:
            state["prompt"] = obs_text

        state["gym_done"] = False
        return state

    async def env_response(self, messages: vf.Messages, state: State, **kwargs) -> vf.Messages:
        env = state["gym_env"]
        
        # 1. Parse Answer (String Extraction) using self.parser
        raw_text = self.parser.parse_answer(messages)
        if raw_text is None:
            # Fallback to raw content if parser fails or returns None
            last_completion = state["trajectory"][-1]["completion"]
            if isinstance(last_completion, list):
                raw_text = str(last_completion[-1]["content"])
            else:
                raw_text = str(last_completion)

        # 2. Parse Action (Type Casting)
        try:
            action = self.action_parser(raw_text)
        except Exception as e:
            state["gym_done"] = True
            state["trajectory"][-1]["reward"] = 0.0
            return [{"role": "user", "content": f"Action Parsing Error: {e}"}]

        # 3. Step Environment
        obs, reward, term, trunc, info = _normalize_step(env.step(action))
        
        # 4. Update State
        state["trajectory"][-1]["reward"] = reward
        state["trajectory"][-1]["extras"]["gym_info"] = info
        state["gym_done"] = term or trunc

        obs_text = self.obs_to_text(obs)
        
        # Append sentinel so the final observability step knows the episode is over
        if state["gym_done"]:
            obs_text = f"{obs_text}\nEpisode already ended."

        if self.message_type == "chat":
            return [{"role": "user", "content": obs_text}]
        return str(obs_text)

    def obs_to_text(self, obs: Any) -> str:
        if self._obs_to_text_fn:
            return self._obs_to_text_fn(obs)
        return str(obs)

    @vf.stop
    async def is_done(self, state: State) -> bool:
        return state.get("gym_done", False)

    @vf.cleanup
    async def cleanup_env(self, state: State):
        env = state.get("gym_env")
        if env and hasattr(env, "close"):
            env.close()

    def _build_auto_dataset(self, n: int, m_type: str) -> Dataset:
        return Dataset.from_dict({
            "example_id": range(n),
            "info": [{"seed": i} for i in range(n)],
            "task": ["gym"] * n,
            "prompt": ["" if m_type=="completion" else []] * n,
            "answer": [""] * n
        })