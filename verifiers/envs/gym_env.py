from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

from datasets import Dataset
from openai import AsyncOpenAI, OpenAI

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
          the dataset row's `info` field. For heterogeneous mode, that is
          the *full* `info` dict (including e.g. `env_type`,
          `env_kwargs`, and anything produced by `info_builder`).
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
    unless 'auto_dummy_eval' is used with heterogeneous mode.
    """
    if message_type == "completion":
        return Dataset.from_dict(
            {
                "prompt": [""],
                "answer": [""],
                "info": [
                    {"env_type": "default", "env_kwargs": {}}
                ],  # For heterogeneous
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
            ],  # For heterogeneous
            "task": ["default"],
            "example_id": [0],
        }
    )


# ---------- GymEnv: Concrete class for Gym-style env *runners* ----------
class GymEnv(Environment):
    """
    Concrete class for running Gym-style RL environments driven by LLMs.

    This class is a "batteries-included" runner that supports three modes:

    1. Homogeneous Mode:
       Pass `env_cls` and `env_kwargs`. Every rollout will use a
       fresh instance of this single env type. The dataset's 'info'
       column is passed to `env.reset()` for initialization (e.g., seeds).

    2. Heterogeneous Mode:
       Pass `env_registry` and a `dataset`. The dataset's 'info'
       column *must* specify the 'env_type' (key in registry)
       and 'env_kwargs' (for __init__) for each rollout.

    3. Custom Mode:
       Do not pass `env_cls` or `env_registry`. Subclass GymEnv
       and implement your own `_make_env(state)` logic.

    ------------------------ Dataset semantics ------------------------

    - At least one of (`dataset`, `eval_dataset`) must be provided, as required
      by `Environment`. GymEnv smooths this in the common RL case by
      auto-generating a dataset when you don't pass one and `auto_dummy_eval`
      is True.

    - Homogeneous mode (env_cls is set, env_registry is None):
        * If you do NOT pass `dataset` / `eval_dataset` and `auto_dummy_eval=True`:
            - `dataset` is auto-generated with `num_train_episodes` rows via
              `_build_auto_dataset(...)`. Each row's `info` dict is passed
              to `env.reset(**info)`.
            - `eval_dataset` is a 1-row dummy dataset (`_default_eval_ds`)
              so that evaluation behaves like "episodes mode" where
              `num_examples` maps naturally onto `rollouts_per_example`.

    - Heterogeneous mode (env_registry is set):
        * If you pass your own dataset:
            - Each row's `info` MUST include:
                - `env_type` (a key in `env_registry`)
                - `env_kwargs` (a dict passed to `env_cls(**env_kwargs)`)
              plus any extra metadata you want; the whole `info` dict is later
              splatted into `reset(**info)`.
        * If you do NOT pass `dataset` / `eval_dataset` and `auto_dummy_eval=True`:
            - We auto-generate a dataset of length `num_train_episodes` via
              `_build_auto_dataset(...)`.
            - For each row `i`, `info.env_type` is chosen in a round-robin over
              `env_registry.keys()`, and `env_kwargs={}` by default, optionally
              augmented by `info_builder(i)`.
            - This auto dataset is used for BOTH `dataset` and `eval_dataset`.
              That means that, in the heterogeneous/no-dataset case, rollouts
              will alternate between the registered envs with default args
              (unless you override via `info_builder`).

    ------------------------ Env interface assumptions ------------------------

    - All env classes used with GymEnv are expected to:
        * Implement `reset(self, **kwargs) -> obs or (obs, info)`:
            - We always pass a dict of keyword arguments coming from the
              dataset row's `info`. For heterogeneous mode, that includes
              things like `env_type`, `env_kwargs`, and any extra fields
              produced by `info_builder`.
        * Implement `step(self, action) -> (obs, reward, done, info)` or
          `(obs, reward, terminated, truncated, info)`.
        * If you register envs in `env_registry`, their `__init__` should
          accept the keys stored in `info["env_kwargs"]`. In practice it is
          often simplest to include a trailing `**kwargs` in your env
          constructor to be robust to extra dataset-driven fields.
    """

    def __init__(
        self,
        *,
        action_parser: Callable[[str], Any],
        # Mode 1: Homogeneous
        env_cls: Optional[type] = None,
        env_kwargs: Dict[str, Any] | None = None,
        # Mode 2: Heterogeneous
        env_registry: Optional[Dict[str, type]] = None,
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
        obs_to_text: Callable[[Any], str] | None = None,
        max_episode_steps: Optional[int] = None,
        rubric: Optional[Rubric] = None,  # defaults to EpisodicSumRubric
        eval_runner: Optional[Callable[..., Any]] = None,
        **env_kwargs_rest: Any,
    ):
        # --- Store env factory info ---
        self._env_cls = env_cls
        self._env_kwargs = dict(env_kwargs or {})
        self._env_registry = env_registry
        self._env_setup_fn = env_setup_fn
        self._info_builder = info_builder
        self._num_train_episodes = num_train_episodes

        # Ensure modes are mutually exclusive
        if self._env_cls and self._env_registry:
            raise ValueError(
                "Cannot provide both 'env_cls' (Homogeneous Mode) and "
                "'env_registry' (Heterogeneous Mode) at the same time."
            )

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
        # dataset plus eval set (homogeneous: dummy eval; heterogeneous: same auto dataset)
        if (
            auto_dummy_eval
            and "dataset" not in final_env_kwargs
            and "eval_dataset" not in final_env_kwargs
        ):
            auto_ds = self._build_auto_dataset(
                num_samples=num_train_episodes,
                message_type=message_type,
            )
            if self._env_registry is not None:
                # Heterogeneous: eval must have valid env_type entries
                # We therefore reuse the auto-generated train dataset so that
                # every row's `info.env_type`/`env_kwargs` is consistent with the registry.
                final_env_kwargs["dataset"] = auto_ds
                final_env_kwargs["eval_dataset"] = auto_ds
            else:
                # Homogeneous: keep episodes-mode semantics via 1-row dummy eval
                final_env_kwargs["dataset"] = auto_ds
                final_env_kwargs["eval_dataset"] = _default_eval_ds(message_type)

        # If eval_dataset is provided but dataset is not, mirror eval->dataset
        # This keeps trainers (which expect env.dataset) happy even if the user
        # only configures an eval dataset.
        if "dataset" not in final_env_kwargs and "eval_dataset" in final_env_kwargs:
            final_env_kwargs["dataset"] = final_env_kwargs["eval_dataset"]

        super().__init__(**final_env_kwargs)

        self.action_parser = action_parser
        self.max_episode_steps = max_episode_steps

        # optional custom evaluator (full override)
        self._eval_runner = eval_runner

        # optional obs_to_text callback; if None, strings are accepted, others error
        self._obs_to_text_fn: Callable[[Any], str] | None = obs_to_text

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
        elif self._env_registry:
            self.env_id = self.env_id or "dataset_driven_env"
            self.env_args = self.env_args or {
                "registry_keys": list(self._env_registry.keys())
            }

    # ----- hooks for concrete envs -----
    def obs_to_text(self, obs: Any) -> str:
        """
        Convert env observation to a string for the LLM.

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

    def _build_initial_prompt(self, obs_text: str) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        if self.few_shot:
            msgs.extend(self.few_shot)
        msgs.append({"role": "user", "content": obs_text})
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

        - Heterogeneous:
            The dataset row's `info` must contain:
              * `env_type` (a key in `self._env_registry`)
              * `env_kwargs` (a dict)
            We then do:
              `env_cls = env_registry[env_type]; env_cls(**env_kwargs)`

            It is therefore recommended that registered env classes have a
            constructor that can safely accept the keys you put into
            `env_kwargs` (including a trailing `**kwargs` if you want to be
            robust to future changes in the dataset).
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
            elif hasattr(env, "setup") and callable(getattr(env, "setup")):
                env.setup()  # type: ignore[attr-defined]

            return env  # type: ignore[return-value]

        # Mode 2: Heterogeneous (env_registry was given)
        if self._env_registry is not None:
            if not state:
                raise ValueError("State is required for Heterogeneous Mode")

            info = state["input"].get("info")
            if not isinstance(info, dict):
                raise ValueError("Dataset 'info' must be a dict for Heterogeneous Mode")

            env_type_name = info.get("env_type")
            if not env_type_name or env_type_name not in self._env_registry:
                raise ValueError(
                    f"Dataset 'info.env_type' missing or '{env_type_name}' "
                    f"not in registry. Available: {list(self._env_registry.keys())}"
                )

            env_cls = self._env_registry[env_type_name]
            env_kwargs = info.get("env_kwargs", {})
            if not isinstance(env_kwargs, dict):
                raise ValueError("Dataset 'info.env_kwargs' must be a dict")

            return env_cls(**env_kwargs)  # type: ignore[return-value]

        # Mode 3: Custom (must be subclassed)
        raise NotImplementedError(
            "GymEnv must be initialized with 'env_cls' (Mode 1) or "
            "'env_registry' (Mode 2), or you must subclass it and "
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

        - Homogeneous mode (no env_registry):
            * Each row's `info` is:
                - `{}` by default, or
                - `info_builder(i)` merged into that dict if `info_builder`
                  is provided.
            * That dict is later splatted into `reset(**info)`.

        - Heterogeneous mode (env_registry is not None):
            * Let `keys = list(env_registry.keys())`.
            * For row `i`, we set:
                - `info["env_type"] = keys[i % len(keys)]`
                - `info["env_kwargs"] = {}`
              so that rows alternate between the registered envs in a simple
              round-robin pattern when a dataset is not explicitly provided.
            * If `info_builder` is provided, its return dict is merged into
              this base `info`. This is the hook you use to:
                - inject per-row `env_kwargs` (e.g. difficulty, seeds), or
                - add extra metadata that `reset(**info)` might care about.

        In all cases:

        - `prompt` is a dummy column ("" or [] depending on message_type) to
          satisfy Environment's expectations; GymEnv ignores it for dynamics.
        - `task` is set to "default".
        - `example_id` is a simple 0..num_samples-1 range so that the trainer
          can shard/group rollouts by example.
        """
        infos: list[dict[str, Any]] = []

        if self._env_registry is not None:
            registry_keys = list(self._env_registry.keys())
            if not registry_keys:
                raise ValueError("env_registry is empty; cannot auto-build dataset")
            for i in range(num_samples):
                env_type_name = registry_keys[i % len(registry_keys)]
                base_info: Dict[str, Any] = {
                    "env_type": env_type_name,
                    "env_kwargs": {},
                }
                if self._info_builder is not None:
                    extra = self._info_builder(i)
                    if not isinstance(extra, dict):
                        raise TypeError("info_builder must return a dict")
                    base_info.update(extra)
                infos.append(base_info)
        else:
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
                "example_id": list(num_samples for _ in range(num_samples)),
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

    # ---- dummy-eval detection ----
    def _is_dummy_eval_ds(self) -> bool:
        """
        Heuristic to detect the built-in dummy eval dataset we create via
        _default_eval_ds(). We only treat that special case as "episodes mode".
        """
        try:
            ds = self.get_eval_dataset(n=-1)
        except Exception:
            return False

        if len(ds) != 1:
            return False

        cols = set(ds.column_names)
        expected = {"prompt", "answer", "info", "task", "example_id"}
        if not expected.issubset(cols):
            return False

        row = ds[0]
        if row.get("task", "default") != "default":
            return False
        if row.get("example_id", 0) != 0:
            return False

        # prompt structure from _default_eval_ds
        prompt = row.get("prompt")
        if self.message_type == "completion":
            if prompt != "":
                return False
        else:  # chat
            if prompt != []:
                return False

        return True

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
        # This call now uses the logic (Homogeneous, Heterogeneous, or Custom)
        env = self._make_env(state)

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

        # In Heterogeneous mode, 'info' was used in _make_env.
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
            history.append({"role": "user", "content": self.obs_to_text(last_obs)})

            # Base prompt for this rollout (mirrors MultiTurnEnv convention)
            state["prompt"] = list(history)

            while not (terminated or truncated) and t < limit:
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
                    state.setdefault("errors", []).append(f"action_parse_error: {e}")
                    truncated = True
                    break

                obs, reward, term, trunc, step_info = _normalize_step(env.step(action))
                t += 1
                terminated, truncated = bool(term), bool(trunc)
                last_obs = obs

                # new observation as user message
                history.append(
                    {"role": "user", "content": self.obs_to_text(last_obs)}
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

            while not (terminated or truncated) and t < limit:
                prompt_text = self.obs_to_text(last_obs)
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
                    state.setdefault("errors", []).append(f"action_parse_error: {e}")
                    truncated = True
                    break

                obs, reward, term, trunc, step_info = _normalize_step(env.step(action))
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

        # Reward is computed later by rubric.score_group from state["trajectory"]
        return state

    # -------- Customizable evaluation layer --------
    async def evaluate(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_results: bool = False,
        save_every: int = -1,
        **kwargs: Any,
    ):
        """
        Evaluate model on this RL environment.

        If a user-supplied eval_runner exists, delegate to it.

        Otherwise:
          - When using the built-in 1-row dummy eval dataset, treat
            num_examples > 1 as "episodes" (map to rollouts_per_example) and
            always set num_examples=1 (explicit RPE still wins).
          - On real datasets, num_examples counts dataset rows
            (capped to dataset size as usual).
        """
        # Full delegation if user supplied their own evaluation runner
        if self._eval_runner is not None:
            maybe = self._eval_runner(
                self,
                client=client,
                model=model,
                sampling_args=sampling_args,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                max_concurrent=max_concurrent,
                max_concurrent_generation=max_concurrent_generation,
                max_concurrent_scoring=max_concurrent_scoring,
                results_path=results_path,
                state_columns=state_columns,
                save_results=save_results,
                save_every=save_every,
                **kwargs,
            )
            if hasattr(maybe, "__await__"):
                return await maybe  # type: ignore[no-any-return]
            return maybe  # type: ignore[no-any-return]

        is_dummy = self._is_dummy_eval_ds()
        try:
            ds = self.get_eval_dataset(n=-1)
            ds_len = len(ds)
        except Exception:
            ds_len = 0
            is_dummy = False

        # Mapping rules:
        # - Dummy:
        #     * If num_examples > 1 and RPE==1 -> set RPE = num_examples, NE = 1
        #     * If num_examples > 1 and RPE>1  -> keep RPE, force NE = 1
        # - Non-dummy: no mapping; NE is capped to dataset size by Environment
        mapped_ne = num_examples
        mapped_rpe = rollouts_per_example
        if is_dummy and num_examples > 1:
            if rollouts_per_example <= 1:
                mapped_rpe = num_examples
            mapped_ne = 1

        results = await super().evaluate(
            client=client,
            model=model,
            sampling_args=sampling_args,
            num_examples=mapped_ne,
            rollouts_per_example=mapped_rpe,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            results_path=results_path,
            state_columns=state_columns,
            save_results=save_results,
            save_every=save_every,
            **kwargs,
        )

        # Correct metadata to reflect the actual evaluation configuration
        try:
            if is_dummy:
                # Always 1 unique example in dummy mode
                results.metadata.num_examples = 1
            else:
                if mapped_ne < 0:
                    results.metadata.num_examples = ds_len
                else:
                    results.metadata.num_examples = min(mapped_ne, ds_len)
            results.metadata.rollouts_per_example = mapped_rpe
        except Exception:
            pass
        return results

    def evaluate_sync(
        self,
        client: OpenAI | AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        results_path: Path | None = None,
        state_columns: list[str] | None = None,
        save_results: bool = False,
        save_every: int = -1,
        **kwargs: Any,
    ):
        """
        Sync wrapper with the same delegation/mapping semantics as the async one.
        """
        # If client is sync OpenAI, adapt to AsyncOpenAI (Environment.generate_sync does this too)
        if isinstance(client, OpenAI):
            client = AsyncOpenAI(api_key=client.api_key, base_url=client.base_url)

        if self._eval_runner is not None:
            maybe = self._eval_runner(
                self,
                client=client,
                model=model,
                sampling_args=sampling_args,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                max_concurrent=max_concurrent,
                max_concurrent_generation=max_concurrent_generation,
                max_concurrent_scoring=max_concurrent_scoring,
                results_path=results_path,
                state_columns=state_columns,
                save_results=save_results,
                save_every=save_every,
                **kwargs,
            )
            if inspect.isawaitable(maybe):
                try:
                    loop = asyncio.get_running_loop()
                    import nest_asyncio  # type: ignore

                    nest_asyncio.apply()
                    return loop.run_until_complete(maybe)  # type: ignore[no-any-return]
                except RuntimeError:
                    return asyncio.run(maybe)  # type: ignore[no-any-return]
            return maybe  # type: ignore[no-any-return]

        is_dummy = self._is_dummy_eval_ds()
        try:
            ds = self.get_eval_dataset(n=-1)
            ds_len = len(ds)
        except Exception:
            ds_len = 0
            is_dummy = False

        mapped_ne = num_examples
        mapped_rpe = rollouts_per_example
        if is_dummy and num_examples > 1:
            if rollouts_per_example <= 1:
                mapped_rpe = num_examples
            mapped_ne = 1

        results = super().evaluate_sync(
            client=client,
            model=model,
            sampling_args=sampling_args,
            num_examples=mapped_ne,
            rollouts_per_example=mapped_rpe,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            results_path=results_path,
            state_columns=state_columns,
            save_results=save_results,
            save_every=save_every,
        )

        try:
            if is_dummy:
                results.metadata.num_examples = 1
            else:
                if mapped_ne < 0:
                    results.metadata.num_examples = ds_len
                else:
                    results.metadata.num_examples = min(mapped_ne, ds_len)
            results.metadata.rollouts_per_example = mapped_rpe
        except Exception:
            pass
        return results
