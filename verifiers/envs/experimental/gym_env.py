from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias, cast

from datasets import Dataset

import verifiers as vf
from verifiers.rubrics.rubric import Rubric
from verifiers.types import MessageType, State


class StepResetEnv(Protocol):
    """
    Protocol for gym-style environments compatible with GymEnv.

    Required methods:
        reset(seed: int) -> obs | (obs, info)
            Reset the environment. Must accept a `seed` keyword argument.
            Returns either just the observation, or (observation, info_dict).

        step(action) -> (obs, reward, done, info) | (obs, reward, done, truncated, info)
            Take an action. Returns a 4-tuple (old gym API) or 5-tuple (new gym API).
            - obs: the new observation
            - reward: float reward for this step
            - done/terminated: bool indicating episode end
            - truncated (optional): bool indicating truncation (defaults to False)
            - info: dict with additional information
    """

    reset: Callable[..., Any]
    step: Callable[..., Any]


ResetOut: TypeAlias = Any | tuple[Any, dict[str, Any]]
StepOut: TypeAlias = (
    tuple[Any, float, bool, bool, dict[str, Any]]
    | tuple[Any, float, bool, dict[str, Any]]
)


def normalize_reset(out: ResetOut) -> tuple[Any, dict[str, Any]]:
    if isinstance(out, tuple) and len(out) == 2:
        return cast(tuple[Any, dict[str, Any]], out)
    return out, {}


def normalize_step(out: StepOut) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    assert isinstance(out, (tuple, list)) and len(out) in (4, 5), (
        f"env.step() returned {type(out)} of length {len(out) if isinstance(out, (tuple, list)) else 'N/A'}, expected tuple of length 4 or 5"
    )
    if len(out) == 5:
        return cast(tuple[Any, float, bool, bool, dict[str, Any]], out)
    obs, reward, done, info = cast(tuple[Any, float, bool, dict[str, Any]], out)
    return obs, float(reward), bool(done), False, info


def sum_step_rewards(state: State) -> float:
    return float(
        sum(
            float(step.get("reward", 0.0) or 0.0)
            for step in state.get("trajectory", [])
        )
    )


class EpisodicSumRubric(Rubric):
    def __init__(self, weight: float = 1.0, **kwargs: Any):
        super().__init__(funcs=[sum_step_rewards], weights=[weight], **kwargs)


class GymEnv(vf.MultiTurnEnv):
    """Universal runner for Gym-compatible environments."""

    def __init__(
        self,
        # gym-specific
        env_cls: type[StepResetEnv],
        env_kwargs: dict[str, Any] | None = None,
        action_parser: Callable[[str], Any] | None = None,
        obs_to_text: Callable[[Any], str] | None = None,
        num_train_episodes: int = 1000,
        num_eval_episodes: int = 20,
        max_episode_steps: int | None = None,
        seed: int = 0,
        # global
        system_prompt: str | None = None,
        few_shot: list[dict[str, Any]] | None = None,
        parser: vf.Parser | None = None,
        rubric: Rubric | None = None,
        message_type: MessageType = "chat",
    ):
        self.env_cls = env_cls
        self.env_kwargs = dict(env_kwargs or {})
        self.action_parser = action_parser or (lambda x: x)
        self.obs_to_text_fn = obs_to_text
        self.num_train_episodes = num_train_episodes
        self.num_eval_episodes = num_eval_episodes
        self.seed = seed
        self.message_type = message_type

        dataset, eval_dataset = self.gym_to_hf()

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            rubric=rubric or EpisodicSumRubric(),
            message_type=message_type,
            max_turns=max_episode_steps or 1000,
            system_prompt=system_prompt,
            few_shot=few_shot,
            parser=parser,
        )

    def gym_to_hf(self) -> tuple[Dataset, Dataset | None]:
        train_rows = []
        eval_rows = []
        total = self.num_train_episodes + self.num_eval_episodes
        env = self.env_cls(**self.env_kwargs)

        try:
            for i in range(total):
                obs, _ = normalize_reset(env.reset(seed=self.seed + i))
                question = self.obs_to_text(obs)
                if self.message_type == "completion":
                    row = {"prompt": question, "answer": str(self.seed + i)}
                else:
                    row = {"question": question, "answer": str(self.seed + i)}
                if i < self.num_train_episodes:
                    train_rows.append(row)
                else:
                    eval_rows.append(row)
        finally:
            close_fn = getattr(env, "close", None)
            if close_fn is not None:
                close_fn()

        dataset = Dataset.from_list(train_rows)
        eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None
        return dataset, eval_dataset

    def obs_to_text(self, obs: Any) -> str:
        """Convert observation to text. Override in subclass for custom formatting."""
        if self.obs_to_text_fn:
            return self.obs_to_text_fn(obs)
        return str(obs)

    def wrap_response(self, text: str) -> vf.Messages:
        if self.message_type == "chat":
            return cast(vf.Messages, [{"role": "user", "content": text}])
        return text

    async def env_response(
        self, messages: vf.Messages, state: State, **kwargs: Any
    ) -> vf.Messages:
        if "gym_env" not in state:
            env = self.env_cls(**self.env_kwargs)
            seed = int(state["answer"])
            env.reset(seed=seed)
            state["gym_env"] = env
            state["gym_done"] = False
        else:
            env = state["gym_env"]

        raw_text = self.parser.parse_answer(messages)
        if raw_text is None:
            last_completion = state["trajectory"][-1]["completion"]
            if isinstance(last_completion, list) and last_completion:
                raw_text = str(last_completion[-1].get("content", ""))
            else:
                raw_text = str(last_completion)

        try:
            action = self.action_parser(raw_text)
        except Exception as e:
            state["gym_done"] = True
            state["trajectory"][-1]["reward"] = 0.0
            err_text = f"Action Parsing Error: {e}"
            return self.wrap_response(err_text)

        obs, reward, term, trunc, info = normalize_step(env.step(action))

        state["trajectory"][-1]["reward"] = reward
        state["trajectory"][-1]["extras"]["gym_info"] = info
        state["gym_done"] = term or trunc

        obs_text = self.obs_to_text(obs)
        if state["gym_done"]:
            obs_text = f"{obs_text}\nEpisode already ended."

        return self.wrap_response(obs_text)

    @vf.stop
    async def is_done(self, state: State) -> bool:
        return state.get("gym_done", False)

    @vf.cleanup
    async def cleanup_env(self, state: State) -> None:
        env = state.pop("gym_env", None)
        if env is not None:
            close_fn = getattr(env, "close", None)
            if close_fn is not None:
                close_fn()
