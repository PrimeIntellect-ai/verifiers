from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

from datasets import Dataset
from pathlib import Path
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
)

# ---------- Protocol: anything with reset/step ----------
ResetOut = Union[Any, Tuple[Any, Dict[str, Any]]]
StepOut = Union[
    Tuple[
        Any, float, bool, bool, Dict[str, Any]
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
            "prompt": [[]],
            "answer": [""],
            "info": [{}],
            "task": ["default"],
            "example_id": [0],
        }
    )


# --- GymEnv ---
class GymEnv(Environment):
    """
    Wrap any env that follows the `step`/`reset` API.

    Subclass and override `obs_to_text` if your env observations aren't already strings.

    Extras:
    - eval_runner: optional user-provided function that fully controls evaluation.
      Signature mirrors `Environment.evaluate`; if set, GymEnv delegates to it.
      If not set, GymEnv uses the default dataset-based evaluation with a small
      quality-of-life tweak: when using the built-in 1-row dummy eval dataset,
      passing num_examples=N behaves like “run N episodes” (i.e., it maps to
      rollouts_per_example=N under the hood).
    """

    def __init__(
        self,
        env: StepResetEnv,
        *,
        action_parser: Callable[[str], Any],
        message_type: MessageType = "chat",
        max_episode_steps: Optional[int] = None,
        setup_fn: Optional[Callable[[StepResetEnv], None]] = None,
        setup_kwargs: Optional[Dict[str, Any]] = None,
        rubric: Optional[
            Rubric
        ] = None,  # you can override; defaults to EpisodicSumRubric
        eval_runner: Optional[Callable[..., Any]] = None,
        **env_kwargs,
    ):
        env_kwargs = dict(env_kwargs)
        env_kwargs["rubric"] = rubric or EpisodicSumRubric()

        # Most Gym-style envs do not need an explicit dataset, as they already contain it
        if "dataset" not in env_kwargs and "eval_dataset" not in env_kwargs:
            env_kwargs["eval_dataset"] = _default_eval_ds(message_type)
        env_kwargs["message_type"] = message_type

        super().__init__(**env_kwargs)

        self.env = env
        self.action_parser = action_parser
        self.max_episode_steps = max_episode_steps
        self._setup_fn = setup_fn
        self._setup_kwargs = setup_kwargs or {}
        self._did_setup = False

        # optional custom evaluator
        self._eval_runner = eval_runner

        self.t = 0
        self.terminated = False
        self.truncated = False
        self.ep_return = 0.0
        self.last_obs: Any = None
        self.last_info: Dict[str, Any] = {}
        self._history_chat: list[ChatMessage] = []
        self._history_text: str = ""

    # --- subclass hook ---
    def obs_to_text(self, obs: Any) -> str:
        """Convert env observation to a string for the chat transcript."""
        if isinstance(obs, str):
            return obs
        raise TypeError(
            "Non-string observation; override obs_to_text() in your subclass."
        )

    # --- internal helper for prompt construction ---
    def _build_initial_prompt(self, obs_text: str) -> list[ChatMessage]:
        msgs: list[ChatMessage] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        if self.few_shot:
            msgs.extend(self.few_shot)
        msgs.append({"role": "user", "content": obs_text})
        return msgs

    # --- setup plumbing ---
    def _maybe_setup(self) -> None:
        if self._did_setup:
            return
        if self._setup_fn:
            self._setup_fn(self.env, **self._setup_kwargs)
        elif hasattr(self.env, "setup") and callable(getattr(self.env, "setup")):
            self.env.setup()  # type: ignore[attr-defined]
        self._did_setup = True

    # --- public step/reset (raw obs returned; text goes to chat only) ---
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        self._maybe_setup()
        obs, info = _normalize_reset(self.env.reset(**kwargs))
        self.t = 0
        self.terminated = False
        self.truncated = False
        self.ep_return = 0.0
        self.last_obs, self.last_info = obs, info

        self._history_chat = []
        self._history_text = ""
        if self.message_type == "chat":
            self._history_chat = self._build_initial_prompt(self.obs_to_text(obs))
        else:
            # for completion mode, callers can seed a non-empty prompt via eval_dataset; otherwise we start empty
            self._history_text = ""
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = _normalize_step(self.env.step(action))
        self.t += 1
        self.ep_return += reward
        self.terminated, self.truncated = term, trunc
        self.last_obs, self.last_info = obs, info

        if self.message_type == "chat":
            self._history_chat.append(
                {"role": "user", "content": self.obs_to_text(obs)}
            )
        return obs, reward, term, trunc, info

    # -------- Environment API: state setup --------
    async def setup_state(self, state: State) -> State:
        """
        GymEnv doesn't need extra state wiring beyond what init_state() does.
        Override if you want to inject env-specific info into state.
        """
        return state

    # -------- Environment API: rollout --------
    async def rollout(
        self,
        input: RolloutInput,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        """
        Run a single RL episode as one rollout.

        We ignore `input["prompt"]` for dynamics; the Gym env itself is the task.
        We still respect the Environment pipeline (init_state, rubric scoring, etc.).
        """
        # Initialize State from RolloutInput
        state = await self.init_state(input, client, model, sampling_args)
        state = await self.setup_state(state)
        state.setdefault("responses", [])
        state["completion"] = None

        # Start a fresh episode every rollout
        _, reset_info = self.reset()

        input_dict = state["input"]
        input_info = input_dict.get("info") if isinstance(input_dict, dict) else None
        if isinstance(input_info, dict):
            merged_info = {**input_info, "reset_info": reset_info}
        else:
            merged_info = {"reset_info": reset_info}
        state["info"] = merged_info

        limit = self.max_episode_steps or getattr(
            self.env, "_max_episode_steps", 10_000
        )

        # Use the oai_tools resolved in init_state (info > env.oai_tools)
        oai_tools = state.get("oai_tools")

        if self.message_type == "chat":
            history: list[ChatMessage] = list(self._history_chat)
            while not (self.terminated or self.truncated) and self.t < limit:
                resp = await self.get_model_response(
                    client=client,
                    model=model,
                    prompt=history,
                    oai_tools=oai_tools,
                    sampling_args=sampling_args or state.get("sampling_args"),
                    message_type="chat",
                )
                assert resp is not None
                text_out = resp.choices[0].message.content or ""  # type: ignore[attr-defined]
                msg: ChatMessage = {"role": "assistant", "content": text_out}
                history.append(msg)

                try:
                    action = self.action_parser(text_out)
                except Exception as e:
                    state.setdefault("errors", []).append(f"action_parse_error: {e}")
                    self.truncated = True
                    break

                obs, reward, term, trunc, step_info = self.step(action)
                state["responses"].append(
                    {
                        "t": self.t,
                        "text": text_out,
                        "action": action,
                        "reward": float(reward),
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                        "info": step_info,
                    }
                )

            # Expose history primarily for debugging/tests
            self._history_chat = list(history)

        else:
            # completion mode: we only feed latest observation text as prompt

            prompt_val = input.get("prompt", "")
            hist_text: str = (
                prompt_val if isinstance(prompt_val, str) else self._history_text
            )
            comp_text: str = ""

            while not (self.terminated or self.truncated) and self.t < limit:
                prompt_text = self.obs_to_text(self.last_obs)
                resp = await self.get_model_response(
                    client=client,
                    model=model,
                    prompt=prompt_text,
                    oai_tools=None,
                    sampling_args=sampling_args or state.get("sampling_args"),
                    message_type="completion",
                )
                assert resp is not None
                text_out = resp.choices[0].text or ""  # type: ignore[attr-defined]
                comp_text = comp_text + text_out if comp_text else text_out

                try:
                    action = self.action_parser(text_out)
                except Exception as e:
                    state.setdefault("errors", []).append(f"action_parse_error: {e}")
                    self.truncated = True
                    break

                obs, reward, term, trunc, step_info = self.step(action)
                state["responses"].append(
                    {
                        "t": self.t,
                        "text": text_out,
                        "action": action,
                        "reward": float(reward),
                        "terminated": bool(term),
                        "truncated": bool(trunc),
                        "info": step_info,
                    }
                )
                hist_text = prompt_text  # next turn uses latest obs only
            self._history_text = hist_text

            if comp_text:
                state["completion"] = comp_text

        # Reward is computed later by rubric.score_group from state["responses"]
        return state

    # expose read-only history for convenience/tests
    @property
    def history(self) -> Messages:
        return self._history_chat if self.message_type == "chat" else self._history_text

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
        **kwargs,
    ):
        """
        If a user-supplied eval_runner exists, delegate to it.
        Otherwise: when using the built-in 1-row dummy eval dataset,
        treat num_examples>1 as "episodes" (map to RPE) and always set
        num_examples=1 (explicit RPE still wins). Also clamp metadata
        num_examples to the actual count used for non-dummy datasets.
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

        # Figure out if we're on the built-in 1-row dummy eval dataset
        try:
            ds = self.get_eval_dataset(n=-1)
            ds_len = len(ds)
            is_dummy_len_one = ds_len == 1
        except Exception:
            ds_len = 0
            is_dummy_len_one = False

        # Mapping rules:
        # - Dummy len==1:
        #     * If num_examples > 1 and RPE==1 -> set RPE = num_examples, NE = 1
        #     * If num_examples > 1 and RPE>1  -> keep RPE, force NE = 1
        # - Non-dummy: no mapping; NE is capped to dataset size
        mapped_ne = num_examples
        mapped_rpe = rollouts_per_example
        if is_dummy_len_one and num_examples > 1:
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
            if is_dummy_len_one:
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
        **kwargs,
    ):
        """
        Sync wrapper with the same delegation/mapping semantics as the async one.
        """
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
            import inspect
            import asyncio

            if inspect.isawaitable(maybe):
                try:
                    loop = asyncio.get_running_loop()
                    import nest_asyncio  # type: ignore

                    nest_asyncio.apply()
                    return loop.run_until_complete(maybe)  # type: ignore[no-any-return]
                except RuntimeError:
                    return asyncio.run(maybe)  # type: ignore[no-any-return]
            return maybe  # type: ignore[no-any-return]

        # Detect dummy vs non-dummy
        try:
            ds = self.get_eval_dataset(n=-1)
            ds_len = len(ds)
            is_dummy_len_one = ds_len == 1
        except Exception:
            ds_len = 0
            is_dummy_len_one = False

        # Apply mapping (see async version for rules)
        mapped_ne = num_examples
        mapped_rpe = rollouts_per_example
        if is_dummy_len_one and num_examples > 1:
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
            **kwargs,
        )

        try:
            if is_dummy_len_one:
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
