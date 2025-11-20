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
    """Sum r_t from state['responses'][i]['reward']."""
    return float(
        sum(float(step.get("reward", 0.0)) for step in state.get("responses", []))
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
    one of (dataset, eval_dataset) is set; the contents do not drive dynamics.
    """
    if message_type == "completion":
        return Dataset.from_dict(
            {
                "prompt": [""],
                "answer": [""],
                "info": [{}],
                "task": ["default"],
                "example_id": [0],
            }
        )
    # chat: prompt is not used, but Environment requires something
    return Dataset.from_dict(
        {
            "prompt": [[]],  # shape matches chat-style prompts, but unused
            "answer": [""],
            "info": [{}],
            "task": ["default"],
            "example_id": [0],
        }
    )


# ---------- GymBaseEnv: base class for RL-style env *runners* ----------
class GymBaseEnv(Environment):
    """
    Base class for Gym-style RL environments driven by LLMs.

    Important: this class is a *runner*, just like Environment itself.
    It does NOT hold per-rollout state. Each call to `rollout()` must be
    self-contained and concurrency-safe.

    You provide:
      - an action_parser: str -> action
      - an implementation of `_make_env(state) -> StepResetEnv`
        which returns a fresh env instance per rollout
      - optionally obs_to_text (or ensure your observations are strings)

    Two usage patterns:

    1. Wrap existing gym-like envs:
       - use GymEnv(env_cls=SomeEnv, env_kwargs=...)
       - maybe pass obs_to_text if obs are not strings

    2. Verifiers-native RL envs:
       - subclass GymBaseEnv
       - implement `_make_env(self, state) -> StepResetEnv`
         returning a tiny object with reset/step and internal state.
    """

    def __init__(
        self,
        *,
        action_parser: Callable[[str], Any],
        # optional datasets (verifiers-style)
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        # if True and no dataset/eval_dataset given, create 1-row dummy eval dataset
        auto_dummy_eval: bool = True,
        message_type: MessageType = "chat",
        obs_to_text: Callable[[Any], str] | None = None,
        max_episode_steps: Optional[int] = None,
        rubric: Optional[Rubric] = None,  # defaults to EpisodicSumRubric
        eval_runner: Optional[Callable[..., Any]] = None,
        **env_kwargs: Any,
    ):
        env_kwargs = dict(env_kwargs)
        env_kwargs["rubric"] = rubric or EpisodicSumRubric()
        env_kwargs["message_type"] = message_type

        # Respect user-provided dataset/eval_dataset if given
        if dataset is not None:
            env_kwargs["dataset"] = dataset
        if eval_dataset is not None:
            env_kwargs["eval_dataset"] = eval_dataset

        # If neither dataset nor eval_dataset, optionally inject dummy eval set
        if (
            auto_dummy_eval
            and "dataset" not in env_kwargs
            and "eval_dataset" not in env_kwargs
        ):
            env_kwargs["eval_dataset"] = _default_eval_ds(message_type)

        super().__init__(**env_kwargs)

        self.action_parser = action_parser
        self.max_episode_steps = max_episode_steps

        # optional custom evaluator (full override)
        self._eval_runner = eval_runner

        # optional obs_to_text callback; if None, strings are accepted, others error
        self._obs_to_text_fn: Callable[[Any], str] | None = obs_to_text

    # ----- hooks for concrete envs -----
    def obs_to_text(self, obs: Any) -> str:
        """
        Convert env observation to a string for the LLM.

        Default behavior:
          - if obs_to_text was passed in __init__, use it
          - else if obs is str, return it
          - else raise: user must define a mapping
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

    # ---- abstract env factory ----
    def _make_env(self, state: State | None = None) -> StepResetEnv:
        """
        Create a fresh env instance for a rollout.

        Must be implemented by subclasses (including GymEnv).
        """
        raise NotImplementedError

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

    # -------- Environment API: state setup --------
    async def setup_state(self, state: State) -> State:
        """
        GymBaseEnv doesn't need extra state wiring beyond what init_state() does.
        Override if you want to inject env-specific info into state, e.g., using
        dataset-provided metadata in state["input"]["info"] to configure reset().
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
        state.setdefault("responses", [])
        state["completion"] = None

        # Fresh env per rollout
        env = self._make_env(state)

        # Episode-local variables
        t = 0
        terminated = False
        truncated = False

        # Start episode
        obs, reset_info = _normalize_reset(env.reset())
        last_obs = obs

        # Merge reset_info with existing input["info"]
        input_dict = state["input"]
        input_info = input_dict.get("info") if isinstance(input_dict, dict) else None
        if isinstance(input_info, dict):
            merged_info = {**input_info, "reset_info": reset_info}
        else:
            merged_info = {"reset_info": reset_info}
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

                state["responses"].append(
                    {
                        "t": t,
                        "text": text_out,
                        "action": action,
                        "reward": float(reward),
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": step_info,
                    }
                )

            # For logging/debugging, expose full conversation if desired
            state["completion"] = history

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

                state["responses"].append(
                    {
                        "t": t,
                        "text": text_out,
                        "action": action,
                        "reward": float(reward),
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": step_info,
                    }
                )

            if comp_text:
                state["completion"] = comp_text

        # Reward is computed later by rubric.score_group from state["responses"]
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


# ---------- GymEnv: env_cls + env_kwargs wrapper ----------
class GymEnv(GymBaseEnv):
    """
    Wrap an external env *by spec* (env_cls + env_kwargs) and create a fresh
    env instance for each rollout. This makes concurrent rollouts safe.

    Usage:
        env = GymEnv(
            env_cls=SomeGymEnv,
            env_kwargs={"arg": 1},
            action_parser=parse_action,
            obs_to_text=lambda obs: f"state: {obs}",
            ...
        )

    You can still pass dataset/eval_dataset if you want verifiers-style control
    via state["input"]["info"], and override _make_env if env construction
    should depend on the dataset row.
    """

    def __init__(
        self,
        env_cls: type,
        env_kwargs: Dict[str, Any] | None = None,
        *,
        action_parser: Callable[[str], Any],
        # optional datasets
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        auto_dummy_eval: bool = True,
        message_type: MessageType = "chat",
        obs_to_text: Callable[[Any], str] | None = None,
        max_episode_steps: Optional[int] = None,
        # env_setup_fn is called on each fresh env instance; can optionally
        # accept (env, state) or just (env)
        env_setup_fn: Optional[Callable[..., None]] = None,
        rubric: Optional[Rubric] = None,
        eval_runner: Optional[Callable[..., Any]] = None,
        **env_kwargs_rest: Any,
    ):
        self._env_cls = env_cls
        self._env_kwargs = dict(env_kwargs or {})
        self._env_setup_fn = env_setup_fn

        super().__init__(
            action_parser=action_parser,
            dataset=dataset,
            eval_dataset=eval_dataset,
            auto_dummy_eval=auto_dummy_eval,
            message_type=message_type,
            obs_to_text=obs_to_text,
            max_episode_steps=max_episode_steps,
            rubric=rubric,
            eval_runner=eval_runner,
            **env_kwargs_rest,
        )

        # Fill env_id/env_args for metadata if user didn't set them
        if not getattr(self, "env_id", ""):
            try:
                self.env_id = getattr(env_cls, "__name__", "gym_env")
            except Exception:
                self.env_id = "gym_env"
        if not getattr(self, "env_args", {}):
            try:
                self.env_args = {"env_cls": str(env_cls), "env_kwargs": self._env_kwargs}
            except Exception:
                self.env_args = {}

    def _make_env(self, state: State | None = None) -> StepResetEnv:
        """Create and (optionally) set up a fresh env instance for this rollout."""
        env = self._env_cls(**self._env_kwargs)  # type: ignore[call-arg]

        # Call user-provided setup if any; support (env, state) or (env)
        if self._env_setup_fn is not None:
            try:
                self._env_setup_fn(env, state)
            except TypeError:
                # user wrote setup_fn(env) only, ignore state
                self._env_setup_fn(env)  # type: ignore[misc]
        elif hasattr(env, "setup") and callable(getattr(env, "setup")):
            env.setup()  # type: ignore[attr-defined]

        return env  # type: ignore[return-value]
