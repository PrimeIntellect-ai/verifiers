from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union
from datasets import Dataset
from pathlib import Path
from openai import AsyncOpenAI, OpenAI
from verifiers.envs.environment import Environment
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State, SamplingArgs, ChatMessage, MessageType

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
      Signature mirrors Environment.evaluate; if set, GymEnv delegates to it.
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
            self._history_chat = self.format_prompt(
                self.obs_to_text(obs), self.system_prompt, self.few_shot
            )
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

    # --- verifiers rollout ---
    async def rollout(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt: Messages,
        completion: Messages | None = None,
        answer: str = "",
        state: State = {},
        task: str = "default",
        info: Dict[str, Any] | None = None,
        example_id: int = 0,
        sampling_args: SamplingArgs | None = None,
        **kwargs,
    ) -> tuple[Messages, State]:
        if self.last_obs is None or self.terminated or self.truncated or self.t == 0:
            self.reset()

        limit = self.max_episode_steps or getattr(
            self.env, "_max_episode_steps", 10_000
        )
        state.setdefault("responses", [])

        if self.message_type == "chat":
            history: list[ChatMessage] = (
                list(prompt)
                if isinstance(prompt, list) and prompt
                else list(self._history_chat)
            )
            comp_msgs: list[ChatMessage] = (
                completion[:] if isinstance(completion, list) else []
            )
            while not (self.terminated or self.truncated) and self.t < limit:
                resp = await self.get_model_response(
                    client=client,
                    model=model,
                    prompt=history,
                    oai_tools=(info or {}).get("oai_tools") if info else None,
                    sampling_args=sampling_args,
                    message_type=self.message_type,
                )
                assert resp is not None
                text_out = resp.choices[0].message.content or ""  # type: ignore[attr-defined]
                msg: ChatMessage = {"role": "assistant", "content": text_out}
                history.append(msg)
                comp_msgs.append(msg)

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
            self._history_chat = list(history)
            return comp_msgs, state

        else:
            # completion mode
            hist_text: str = (
                prompt
                if isinstance(prompt, str) and len(prompt) > 0
                else self._history_text
            )
            comp_text: str = completion if isinstance(completion, str) else ""

            while not (self.terminated or self.truncated) and self.t < limit:
                # For completion mode, we feed only the latest observation text as the prompt.
                # If caller wants accumulating prompts, they can subclass and override this method.
                prompt_text = self.obs_to_text(self.last_obs)
                resp = await self.get_model_response(
                    client=client,
                    model=model,
                    prompt=prompt_text,
                    oai_tools=None,
                    sampling_args=sampling_args,
                    message_type=self.message_type,
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
            if self.message_type == "completion":
                joined = "\n".join(
                    [r["text"] for r in state.get("responses", []) if "text" in r]
                )
                return joined, state
            return comp_text, state

    # expose read-only history for convenience/tests
    @property
    def history(self) -> Messages:
        return self._history_chat if self.message_type == "chat" else ""

    # -------- Customizable evaluation layer --------
    async def evaluate(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: SamplingArgs | None = None,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
        results_path: "Path | None" = None,
        state_columns: "list[str] | None" = None,
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
        if self._eval_runner is not None:
            # Let user fully own evaluation. We pass through all args.
            maybe = self._eval_runner(
                self,
                client=client,
                model=model,
                sampling_args=sampling_args,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                score_rollouts=score_rollouts,
                max_concurrent=max_concurrent,
                max_concurrent_generation=max_concurrent_generation,
                max_concurrent_scoring=max_concurrent_scoring,
                interleave_scoring=interleave_scoring,
                results_path=results_path,
                state_columns=state_columns,
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
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            interleave_scoring=interleave_scoring,
            results_path=results_path,
            state_columns=state_columns,
            save_every=save_every,
            **kwargs,
        )

        # Correct metadata to reflect the actual evaluation configuration
        try:
            if is_dummy_len_one:
                # Always 1 unique example in dummy mode
                results.metadata.num_examples = 1
            else:
                # Clamp to the real dataset size if requested NE was larger
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
        score_rollouts: bool = True,
        max_concurrent: int = -1,
        max_concurrent_generation: int | None = None,
        max_concurrent_scoring: int | None = None,
        interleave_scoring: bool = True,
        results_path: "Path | None" = None,
        state_columns: "list[str] | None" = None,
        save_every: int = -1,
        **kwargs,
    ):
        """
        Sync wrapper with the same delegation/mapping semantics as the async one.
        """
        if self._eval_runner is not None:
            maybe = self._eval_runner(
                self,
                client=client,  # pass through unchanged
                model=model,
                sampling_args=sampling_args,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
                score_rollouts=score_rollouts,
                max_concurrent=max_concurrent,
                max_concurrent_generation=max_concurrent_generation,
                max_concurrent_scoring=max_concurrent_scoring,
                interleave_scoring=interleave_scoring,
                results_path=results_path,
                state_columns=state_columns,
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
            score_rollouts=score_rollouts,
            max_concurrent=max_concurrent,
            max_concurrent_generation=max_concurrent_generation,
            max_concurrent_scoring=max_concurrent_scoring,
            interleave_scoring=interleave_scoring,
            results_path=results_path,
            state_columns=state_columns,
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
