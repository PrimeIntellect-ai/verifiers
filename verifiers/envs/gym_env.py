from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, List

from datasets import Dataset
from openai import AsyncOpenAI

from verifiers.envs.environment import Environment
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    Messages,
    State,
    SamplingArgs,
    ChatMessage,
    MessageType,
    RolloutInput,
    TrajectoryStep,
)
from verifiers.utils.response_utils import (
    parse_response_messages,
    parse_response_tokens,
)
from verifiers.utils.message_utils import concat_messages

# ---------- Protocol: anything with reset/step ----------
ResetOut = Union[Any, Tuple[Any, Dict[str, Any]]]
StepOut = Union[
    Tuple[
        Any,
        float,
        bool,
        bool,
        Dict[str, Any],
    ],  # (obs, reward, terminated, truncated, info)
    Tuple[Any, float, bool, Dict[str, Any]],  # (obs, reward, done, info)
]


class StepResetEnv(Protocol):
    """
    Minimal protocol for GymEnv-compatible environments.

    The key assumptions are:

    - `reset(**kwargs)`:
        * Must accept **kwargs, because we always splat a dict coming from
          the dataset row's `info` field.
        * May return either:
            - `obs`
            - `(obs, info_dict)`
          where `info_dict` is merged into the rollout state's `info`.

    - `step(action)`:
        * Must return either:
            - `(obs, reward, done, info)`
            - `(obs, reward, terminated, truncated, info)`
          which is then normalized into a 5-tuple internally.
    """
    def reset(self, **kwargs) -> ResetOut: ...
    def step(self, action: Any) -> StepOut: ...


def _normalize_reset(out: ResetOut) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(out, tuple) and len(out) == 2:
        return out  # (obs, info)
    return out, {}  # obs only


def _normalize_step(out: StepOut) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    if not isinstance(out, (tuple, list)):
        raise TypeError(
            "env.step(action) must return a tuple/list of length 4 or 5, got "
            f"{type(out)}"
        )
    if len(out) == 5:
        obs, reward, term, trunc, info = out
        return obs, float(reward), bool(term), bool(trunc), info
    elif len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, info
    raise RuntimeError("env.step(action) must return 4 or 5 elements.")


# ---------- Episodic rubric (sums stepwise rewards) ----------
def sum_step_rewards(*, state: State, **_) -> float:
    """Sum r_t from state['trajectory'][i]['reward']."""
    return float(
        sum(float(step.get("reward", 0.0)) for step in state.get("trajectory", []))
    )


class EpisodicSumRubric(Rubric):
    """Default rubric: sum per-step rewards."""

    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(funcs=[sum_step_rewards], weights=[weight], **kwargs)


# --- tiny helper to avoid forcing users to pass a dataset ---
def _default_eval_ds(message_type: str) -> Dataset:
    """
    Built-in dummy eval dataset used when the user doesn't provide one.

    We only ever use this to satisfy Environment's requirement that at least
    one of (dataset, eval_dataset) is set; the contents do not drive dynamics
    unless 'auto_dummy_eval' is used.
    """
    if message_type == "completion":
        return Dataset.from_dict(
            {
                "prompt": [""],
                "answer": [""],
                "info": [
                    {"env_type": "default", "env_kwargs": {}}
                ],
                "task": ["default"],
                "example_id": [0],
            }
        )
    # chat: prompt is not used, but Environment requires something
    return Dataset.from_dict(
        {
            "prompt": [[]],  # shape matches chat-style prompts, but unused
            "answer": [""],
            "info": [
                {"env_type": "default", "env_kwargs": {}}
            ],
            "task": ["default"],
            "example_id": [0],
        }
    )


# ---------- GymEnv: Concrete class for Gym-style env *runners* ----------
class GymEnv(Environment):
    """
    Concrete class for running Gym-style RL environments driven by LLMs.

    This class is a "batteries-included" runner that supports two modes:

    1. Homogeneous Mode:
       Pass `env_cls` and `env_kwargs`. Every rollout will use a
       fresh instance of this single env type. The dataset's 'info'
       column is passed to `env.reset()` for initialization (e.g., seeds).

    2. Custom Mode:
       Do not pass `env_cls`. Subclass GymEnv
       and implement your own `_make_env(state)` logic.

    ------------------------ Dataset semantics ------------------------

    - At least one of (`dataset`, `eval_dataset`) must be provided, as required
      by `Environment`. GymEnv smooths this in the common RL case by
      auto-generating a dataset when you don't pass one and `auto_dummy_eval`
      is True.

    - Homogeneous mode (env_cls is set):
        * If you do NOT pass `dataset` / `eval_dataset` and `auto_dummy_eval=True`:
            - `dataset` is auto-generated with `num_train_episodes` rows via
              `_build_auto_dataset(...)`. Each row's `info` dict is passed
              to `env.reset(**info)`.
            - `eval_dataset` is a 1-row dummy dataset (`_default_eval_ds`)
              so that evaluation behaves like "episodes mode" where
              `num_examples` maps naturally onto `rollouts_per_example`.

        * If you pass ONLY `eval_dataset` (no `dataset`) and `auto_dummy_eval=True`:
            - `eval_dataset` is used *only* for evaluation.
            - A separate train `dataset` is auto-generated with
              `num_train_episodes` rows via `_build_auto_dataset(...)`.
              We deliberately do **not** mirror eval into train.

    ------------------------ Env interface assumptions ------------------------

    - All env classes used with GymEnv are expected to:
        * Implement `reset(self, **kwargs) -> obs or (obs, info)`:
            - We always pass a dict of keyword arguments coming from the
              dataset row's `info`.
        * Implement `step(self, action) -> (obs, reward, done, info)` or
          `(obs, reward, terminated, truncated, info)`.
    """

    def __init__(
        self,
        *,
        action_parser: Callable[[str], Any],
        # Mode 1: Homogeneous
        env_cls: Optional[type] = None,
        env_kwargs: Dict[str, Any] | None = None,
        # env_setup_fn is for Homogeneous mode (Mode 1)
        env_setup_fn: Optional[Callable[..., None]] = None,
        # optional datasets (verifiers-style)
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        # if True and no dataset/eval_dataset given, create 1-row dummy eval dataset
        auto_dummy_eval: bool = True,
        num_train_episodes: int = 1024,
        info_builder: Optional[Callable[[int], Dict[str, Any]]] = None,
        message_type: MessageType = "chat",
        # obs_to_text can return str or List[dict] for multimodal
        obs_to_text: Callable[[Any], str | List[Dict[str, Any]]] | None = None,
        max_episode_steps: Optional[int] = None,
        rubric: Optional[Rubric] = None,  # defaults to EpisodicSumRubric
        **env_kwargs_rest: Any,
    ):
        # --- Store env factory info ---
        self._env_cls = env_cls
        self._env_kwargs = dict(env_kwargs or {})
        self._env_setup_fn = env_setup_fn
        self._info_builder = info_builder
        self._num_train_episodes = num_train_episodes

        # --- Prepare args for Environment.__init__ ---
        final_env_kwargs = dict(env_kwargs_rest)
        final_env_kwargs["rubric"] = rubric or EpisodicSumRubric()
        final_env_kwargs["message_type"] = message_type

        # Respect user-provided dataset/eval_dataset if given
        if dataset is not None:
            final_env_kwargs["dataset"] = dataset
        if eval_dataset is not None:
            final_env_kwargs["eval_dataset"] = eval_dataset

        # If neither dataset nor eval_dataset, optionally inject auto-generated train
        # dataset plus dummy eval set
        if (
            auto_dummy_eval
            and "dataset" not in final_env_kwargs
            and "eval_dataset" not in final_env_kwargs
        ):
            auto_ds = self._build_auto_dataset(
                num_samples=num_train_episodes,
                message_type=message_type,
            )
            # Homogeneous: keep episodes-mode semantics via 1-row dummy eval
            final_env_kwargs["dataset"] = auto_ds
            final_env_kwargs["eval_dataset"] = _default_eval_ds(message_type)

        # If the user passes only eval_dataset (no dataset), we do NOT mirror
        # eval into train (that would be a surprising leakage). In homogeneous
        # mode we can still auto-build a separate train dataset when requested.
        if (
            auto_dummy_eval
            and "dataset" not in final_env_kwargs
            and "eval_dataset" in final_env_kwargs
        ):
            final_env_kwargs["dataset"] = self._build_auto_dataset(
                num_samples=num_train_episodes,
                message_type=message_type,
            )

        super().__init__(**final_env_kwargs)

        self.action_parser = action_parser
        self.max_episode_steps = max_episode_steps

        # optional obs_to_text callback; if None, strings are accepted, others error
        self._obs_to_text_fn = obs_to_text

        # --- Metadata (moved from old GymEnv) ---
        if self._env_cls:
            if not getattr(self, "env_id", ""):
                try:
                    self.env_id = getattr(self._env_cls, "__name__", "gym_env")
                except Exception:
                    self.env_id = "gym_env"
            if not getattr(self, "env_args", {}):
                try:
                    self.env_args = {
                        "env_cls": str(self._env_cls),
                        "env_kwargs": self._env_kwargs,
                    }
                except Exception:
                    self.env_args = {}

    # ----- hooks for concrete envs -----
    def obs_to_text(self, obs: Any) -> str | List[Dict[str, Any]]:
        """
        Convert env observation to a string (or multimodal list) for the LLM.

        Default behavior:
          - if obs_to_text was passed in __init__, use it
          - else if obs is str, return it
          - else raise: user must define a mapping

        In other words, for non-string observations you must either:
          * pass `obs_to_text=...` when constructing GymEnv, or
          * subclass GymEnv and override `obs_to_text()`.
        """
        if self._obs_to_text_fn is not None:
            return self._obs_to_text_fn(obs)
        if isinstance(obs, str):
            return obs
        raise TypeError(
            "Non-string observation and no obs_to_text provided; "
            "either pass obs_to_text=... in __init__ or override obs_to_text()."
        )

    def _build_initial_prompt(
        self, obs_content: str | List[Dict[str, Any]]
    ) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        if self.few_shot:
            msgs.extend(self.few_shot)
        # Type ignore: chat messages can technically support lists for multimodal
        msgs.append({"role": "user", "content": obs_content})  # type: ignore
        return msgs

    # ---- concrete env factory ----
    def _make_env(self, state: State | None = None) -> StepResetEnv:
        """
        Create a fresh env instance for a rollout based on __init__ mode.

        This method can be overridden by subclasses for custom env creation.

        Contracts:

        - Homogeneous:
            `env_cls` must construct an object implementing StepResetEnv.
            The constructor is called as: `env_cls(**self._env_kwargs)`.
        """
        # Mode 1: Homogeneous (env_cls was given)
        if self._env_cls is not None:
            env = self._env_cls(**self._env_kwargs)  # type: ignore[call-attr]

            # Call user-provided setup if any
            if self._env_setup_fn is not None:
                try:
                    self._env_setup_fn(env, state)
                except TypeError:
                    self._env_setup_fn(env)  # type: ignore[misc]
            # Safely check if setup exists before calling
            elif hasattr(env, "setup") and callable(getattr(env, "setup")):
                env.setup()  # type: ignore[attr-defined]

            return env  # type: ignore[return-value]

        # Mode 3: Custom (must be subclassed)
        raise NotImplementedError(
            "GymEnv must be initialized with 'env_cls' (Mode 1), "
            "or you must subclass it and "
            "implement your own _make_env(state) method."
        )

    def _episode_limit_for_env(self, env: Any) -> int:
        """
        Determine maximum episode length given an env instance.

        Preference order:
          1. explicit self.max_episode_steps
          2. env._max_episode_steps
          3. env.spec.max_episode_steps (gym/gymnasium-style)
          4. fallback 10_000
        """
        if self.max_episode_steps is not None:
            return self.max_episode_steps

        limit = getattr(env, "_max_episode_steps", None)
        if isinstance(limit, int) and limit > 0:
            return limit

        spec = getattr(env, "spec", None)
        if spec is not None:
            spec_limit = getattr(spec, "max_episode_steps", None)
            if isinstance(spec_limit, int) and spec_limit > 0:
                return spec_limit

        return 10_000

    # ---- auto dataset generation ----
    def _build_auto_dataset(
        self,
        num_samples: int,
        message_type: MessageType,
    ) -> Dataset:
        """
        Build a simple dataset of "hidden state" descriptors, used when the user
        doesn't provide a dataset. The `info` per row describes how to init the env.

        - Homogeneous mode:
            * Each row's `info` is:
                - `{}` by default, or
                - `info_builder(i)` merged into that dict if `info_builder`
                  is provided.
            * That dict is later splatted into `reset(**info)`.

        In all cases:

        - `prompt` is a dummy column ("" or [] depending on message_type) to
          satisfy Environment's expectations; GymEnv ignores it for dynamics.
        - `task` is set to "default".
        - `example_id` is a simple 0..num_samples-1 range so that the trainer
          can shard/group rollouts by example.
        """
        infos: list[dict[str, Any]] = []

        for i in range(num_samples):
            base_info: Dict[str, Any] = {}
            if self._info_builder is not None:
                extra = self._info_builder(i)
                if not isinstance(extra, dict):
                    raise TypeError("info_builder must return a dict")
                base_info.update(extra)
            infos.append(base_info)

        if message_type == "completion":
            prompts = ["" for _ in range(num_samples)]
        else:
            prompts = [[] for _ in range(num_samples)]

        return Dataset.from_dict(
            {
                "prompt": prompts,
                "answer": ["" for _ in range(num_samples)],
                "info": infos,
                "task": ["default" for _ in range(num_samples)],
                "example_id": list(range(num_samples)),
            }
        )

    # -------- Environment API: state setup --------
    async def setup_state(self, state: State) -> State:
        """
        GymEnv doesn't need extra state wiring beyond what init_state() does.
        The dataset 'info' is used later, inside `rollout()`->`_make_env()`
        and `rollout()`->`env.reset()`.
        """
        return state

    async def _add_trajectory_step(
        self,
        state: State,
        prompt_messages: Messages,
        response,
    ) -> TrajectoryStep | None:
        """
        Convert a raw model response into a TrajectoryStep and append to state.
        Returns the step or None if overlong-prompt.
        """
        if response is not None and getattr(response, "id", None) == "overlong-prompt":
            state["prompt_too_long"] = True
            return None

        completion_messages = await parse_response_messages(
            response, self.message_type
        )
        tokens = await parse_response_tokens(
            response, self.message_type, self.max_seq_len
        )
        step: TrajectoryStep = TrajectoryStep(
            prompt=prompt_messages,
            completion=completion_messages,
            response=response,
            tokens=tokens,
            reward=None,
            advantage=None,
            extras={},
        )
        state["trajectory"].append(step)
        return step

    # -------- Environment API: rollout --------
    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Run a single RL episode as one rollout, using a *fresh* env instance
        returned by `_make_env(state)`.

        This is concurrency-safe: each async rollout gets its own env.
        No per-rollout state is stored on self.
        """
        # Initialize State from RolloutInput
        state = await self.init_state(input, client, model, sampling_args)
        state = await self.setup_state(state)
        state["completion"] = None

        # Fresh env per rollout.
        # This call now uses the logic (Homogeneous or Custom)
        env = self._make_env(state)

        # Wrap in try/finally to ensure proper cleanup if env.close() exists
        try:
            # Episode-local variables
            t = 0
            terminated = False
            truncated = False

            # --- Start episode ---
            # We pass dataset info to reset()
            # (e.g., info={'seed': 123} -> env.reset(seed=123))
            # This is the primary way the dataset "initializes" the env.
            input_dict = state["input"]
            input_info = input_dict.get("info") if isinstance(input_dict, dict) else {}
            if not isinstance(input_info, dict):
                input_info = {}

            # In Homogeneous mode, 'info' is used here in reset().
            obs, reset_info = _normalize_reset(env.reset(**input_info))
            last_obs = obs

            # Merge reset_info with existing input["info"]
            merged_info = {**input_info, "reset_info": reset_info}
            state["info"] = merged_info

            limit = self._episode_limit_for_env(env)

            # Use the oai_tools resolved in init_state (info > env.oai_tools)
            oai_tools = state.get("oai_tools")

            if self.message_type == "chat":
                history: list[ChatMessage] = []
                if self.system_prompt:
                    history.append({"role": "system", "content": self.system_prompt})
                if self.few_shot:
                    history.extend(self.few_shot)

                # initial observation
                # cast logic: we trust obs_to_text returns something chat-friendly (str or list)
                history.append(
                    {"role": "user", "content": self.obs_to_text(last_obs)}  # type: ignore
                )

                # Base prompt for this rollout (mirrors MultiTurnEnv convention)
                state["prompt"] = list(history)

                # We obey:
                # 1. Gym flags (terminated, truncated)
                # 2. Episode limit (t < limit)
                # 3. Environment/Verifiers stops (is_completed)
                while (
                    not (terminated or truncated)
                    and t < limit
                    and not await self.is_completed(state)
                ):
                    prompt_messages: Messages = list(history)

                    resp = await self.get_model_response(
                        client=client,
                        model=model,
                        prompt=prompt_messages,
                        oai_tools=oai_tools,
                        sampling_args=sampling_args or state.get("sampling_args"),
                        message_type="chat",
                    )
                    assert resp is not None

                    step = await self._add_trajectory_step(state, prompt_messages, resp)
                    if step is None:
                        # overlong prompt, bailout
                        truncated = True
                        break

                    # for action parsing, use the raw response content
                    text_out = resp.choices[0].message.content or ""  # type: ignore[attr-defined]
                    history.append({"role": "assistant", "content": text_out})

                    try:
                        action = self.action_parser(text_out)
                    except Exception as e:
                        state.setdefault("errors", []).append(
                            f"action_parse_error: {e}"
                        )
                        truncated = True
                        break

                    obs, reward, term, trunc, step_info = _normalize_step(
                        env.step(action)
                    )
                    t += 1
                    terminated, truncated = bool(term), bool(trunc)
                    last_obs = obs

                    # new observation as user message
                    history.append(
                        {"role": "user", "content": self.obs_to_text(last_obs)}  # type: ignore
                    )

                    # Update the TrajectoryStep with RL info
                    step["reward"] = float(reward)
                    step["extras"]["action"] = action
                    step["extras"]["text_out"] = text_out
                    step["extras"]["terminated"] = terminated
                    step["extras"]["truncated"] = truncated
                    step["extras"]["step_info"] = step_info
                    step["extras"]["t"] = t

                # For logging/debugging, expose full conversation if desired
                if state["trajectory"]:
                    last_step = state["trajectory"][-1]
                    last_prompt = last_step["prompt"]
                    last_completion = last_step["completion"]
                    full_conversation = concat_messages([last_prompt, last_completion])
                    base_prompt = state.get("prompt", [])
                    if isinstance(base_prompt, list):
                        state["completion"] = full_conversation[len(base_prompt) :]
                    else:
                        state["completion"] = full_conversation
                else:
                    state["completion"] = []

            else:
                # completion mode: prompt is always the latest observation text
                comp_text: str = ""
                # local last_obs already set from reset

                while (
                    not (terminated or truncated)
                    and t < limit
                    and not await self.is_completed(state)
                ):
                    # For completion mode, we assume obs_to_text returns a simple str
                    prompt_text = str(self.obs_to_text(last_obs))
                    prompt_messages: Messages = prompt_text  # completion API uses str

                    resp = await self.get_model_response(
                        client=client,
                        model=model,
                        prompt=prompt_text,
                        oai_tools=None,
                        sampling_args=sampling_args or state.get("sampling_args"),
                        message_type="completion",
                    )
                    assert resp is not None

                    step = await self._add_trajectory_step(state, prompt_messages, resp)
                    if step is None:
                        truncated = True
                        break

                    text_out = resp.choices[0].text or ""  # type: ignore[attr-defined]
                    comp_text = comp_text + text_out if comp_text else text_out

                    try:
                        action = self.action_parser(text_out)
                    except Exception as e:
                        state.setdefault("errors", []).append(
                            f"action_parse_error: {e}"
                        )
                        truncated = True
                        break

                    obs, reward, term, trunc, step_info = _normalize_step(
                        env.step(action)
                    )
                    t += 1
                    terminated, truncated = bool(term), bool(trunc)
                    last_obs = obs

                    # Update the TrajectoryStep with RL info
                    step["reward"] = float(reward)
                    step["extras"]["action"] = action
                    step["extras"]["text_out"] = text_out
                    step["extras"]["terminated"] = terminated
                    step["extras"]["truncated"] = truncated
                    step["extras"]["step_info"] = step_info
                    step["extras"]["t"] = t

                if comp_text:
                    state["completion"] = comp_text

            # --- Post-Rollout Metrics Extraction ---
            # Automatically lift scalar values from the FINAL step's info dict into state["metrics"].
            if state["trajectory"]:
                final_step = state["trajectory"][-1]
                final_info = final_step["extras"].get("step_info", {})

                if "metrics" not in state or state["metrics"] is None:
                    state["metrics"] = {}

                for k, v in final_info.items():
                    if isinstance(v, (int, float, bool)):
                        state["metrics"][k] = float(v)

        finally:
            # Gracefully handle cleanup if the environment supports it
            if hasattr(env, "close") and callable(getattr(env, "close")):
                env.close()  # type: ignore

        # Reward is computed later by rubric.score_group from state["trajectory"]
        return state