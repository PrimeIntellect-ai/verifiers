"""Lightweight monkey-patching integration for Braintrust tracing in verifiers.

Instead of importing modified environment classes from the experimental folder,
users can activate tracing with a single call at the top of their environment::

    from verifiers.envs.experimental.braintrust_tracing.integration import setup_verifiers_tracing
    setup_verifiers_tracing()

This patches the core ``Environment``, ``MultiTurnEnv``, and ``ToolEnv`` classes
in-place so that all subclasses (including user-defined environments) automatically
produce Braintrust traces without any other code changes.

The patching is idempotent — calling ``setup_verifiers_tracing()`` multiple times
is safe and has no additional effect after the first call.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import verifiers.envs.experimental.braintrust_tracing.braintrust_tracing as _bt
from verifiers.utils.usage_utils import extract_usage_tokens

_log = logging.getLogger(__name__)

_PATCH_MARKER = "__verifiers_bt_patched__"


def _is_patched(cls: type) -> bool:
    return cls.__dict__.get(_PATCH_MARKER, False)


def _mark_patched(cls: type) -> None:
    setattr(cls, _PATCH_MARKER, True)


# ---------------------------------------------------------------------------
# Environment patches
# ---------------------------------------------------------------------------


def _patch_environment() -> None:
    from verifiers.envs.environment import Environment

    if _is_patched(Environment):
        return

    _orig_get_model_response = Environment.get_model_response

    async def _traced_get_model_response(
        self: Any,
        state: Any,
        prompt: Any,
        client: Any = None,
        model: Any = None,
        tool_defs: Any = None,
        sampling_args: Any = None,
    ) -> Any:
        resolved_model = model or state.get("model", "")

        bt_parent = state.get("_bt_turn_span") or state.get("_bt_span")
        bt_span = _bt.model_request_span(
            bt_parent,
            model=resolved_model,
            turn_index=len(state.get("trajectory", [])),
            messages=prompt,
        )
        t0 = time.monotonic()
        error_msg = ""
        response = None
        try:
            response = await _orig_get_model_response(
                self, state, prompt, client, model, tool_defs, sampling_args
            )
        except Exception as exc:
            error_msg = repr(exc)[:500]
            raise
        except BaseException as exc:
            error_msg = repr(exc)[:500]
            raise
        finally:
            dur = time.monotonic() - t0
            input_tok, output_tok = 0.0, 0.0
            if not error_msg and response is not None:
                input_tok, output_tok = (
                    float(v) for v in extract_usage_tokens(response)
                )
            _bt.model_request_completed(
                bt_span,
                duration_s=dur,
                input_tokens=input_tok,
                output_tokens=output_tok,
                response=response if not error_msg else None,
                error=error_msg,
            )
        return response

    async def _traced_run_rollout_state(
        self: Any,
        input: Any,
        client: Any,
        model: str,
        sampling_args: Any,
    ) -> Any:
        t0 = time.monotonic()
        bt_span = _bt.rollout_started(
            env_id=getattr(self, "env_id", ""),
            model=model,
            example_id=input.get("example_id", ""),
            trajectory_id="",
        )
        _bt._pending_rollout_span.set(bt_span)

        state = await self.rollout(input, client, model, sampling_args)

        bt_score = _bt.scoring_started(
            bt_span, trajectory_id=state.get("trajectory_id", "")
        )
        state["timing"].scoring.start = time.time()
        if self.score_rollouts:
            await self.rubric.score_rollout(state)
        else:
            await self.rubric.dummy_score_rollout(state)
        state["timing"].scoring.end = time.time()
        scoring_dur = state["timing"].scoring.end - state["timing"].scoring.start
        _bt.scoring_completed(
            bt_score, duration_s=scoring_dur, reward=state.get("reward")
        )

        await self.rubric.cleanup(state)

        usage = self.get_state_usage(state) or {}
        _bt.rollout_completed(
            bt_span,
            reward=state.get("reward"),
            num_turns=len(state.get("trajectory", [])),
            duration_s=time.monotonic() - t0,
            stop_condition=state.get("stop_condition", ""),
            error=repr(state["error"])[:500] if state.get("error") else "",
            input_tokens=float(usage.get("input_tokens", 0)),
            output_tokens=float(usage.get("output_tokens", 0)),
        )
        return state

    async def _traced_run_group_states(
        self: Any,
        group_inputs: list,
        client: Any,
        model: str,
        sampling_args: Any,
    ) -> list:
        t0 = time.monotonic()
        example_id = group_inputs[0].get("example_id", "") if group_inputs else ""
        bt_group = _bt.group_started(
            env_id=getattr(self, "env_id", ""),
            model=model,
            example_id=example_id,
            group_size=len(group_inputs),
        )

        bt_rollout_spans: list[object | None] = [None for _ in group_inputs]
        rollout_start_times: list[float] = [0.0] * len(group_inputs)

        async def _traced_rollout(idx: int, ri: Any) -> Any:
            r_t0 = time.monotonic()
            bt_rollout = _bt.start_child_span(
                bt_group,
                name="rollout",
                span_type="task",
                input={"example_id": _bt._safe(ri.get("example_id", ""))},
                metadata={"env_id": getattr(self, "env_id", ""), "model": model},
            )
            bt_rollout_spans[idx] = bt_rollout
            rollout_start_times[idx] = r_t0
            _bt._pending_rollout_span.set(bt_rollout)
            return await self.rollout(ri, client, model, sampling_args)

        group_states = await asyncio.gather(
            *[_traced_rollout(i, inp) for i, inp in enumerate(group_inputs)]
        )

        start_scoring = time.time()
        for state in group_states:
            state["timing"].scoring.start = start_scoring
        if self.score_rollouts:
            await self.rubric.score_group(group_states)
        else:
            await self.rubric.dummy_score_group(group_states)
        end_scoring = time.time()
        for state in group_states:
            state["timing"].scoring.end = end_scoring

        for state in group_states:
            await self.rubric.cleanup(state)

        now = time.monotonic()
        for st, bt_rollout, r_t0 in zip(
            group_states, bt_rollout_spans, rollout_start_times
        ):
            usage = self.get_state_usage(st) or {}
            _bt.rollout_completed(
                bt_rollout,
                reward=st.get("reward"),
                num_turns=len(st.get("trajectory", [])),
                duration_s=now - r_t0,
                stop_condition=st.get("stop_condition", ""),
                error=repr(st["error"])[:500] if st.get("error") else "",
                input_tokens=float(usage.get("input_tokens", 0)),
                output_tokens=float(usage.get("output_tokens", 0)),
            )

        rewards = [s.get("reward") for s in group_states if s.get("reward") is not None]
        _bt.group_completed(
            bt_group,
            duration_s=time.monotonic() - t0,
            avg_reward=sum(rewards) / len(rewards) if rewards else None,
            group_size=len(group_inputs),
        )
        return group_states

    Environment.get_model_response = _traced_get_model_response  # type: ignore[assignment]
    Environment._run_rollout_state = _traced_run_rollout_state  # type: ignore[assignment]
    Environment._run_group_states = _traced_run_group_states  # type: ignore[assignment]
    _mark_patched(Environment)


# ---------------------------------------------------------------------------
# MultiTurnEnv patches
# ---------------------------------------------------------------------------


def _patch_multiturn_env() -> None:
    from verifiers.envs.multiturn_env import MultiTurnEnv

    if _is_patched(MultiTurnEnv):
        return

    async def _traced_rollout(
        self: Any,
        input: Any,
        client: Any,
        model: str,
        sampling_args: Any = None,
    ) -> Any:
        import verifiers as vf
        from verifiers.types import TimeSpan
        from verifiers.utils.message_utils import maybe_normalize_messages

        state = await self.init_state(input, client, model, sampling_args)
        _env_id = getattr(self, "env_id", "")

        bt_rollout = _bt._pending_rollout_span.get(None)
        if bt_rollout is not None:
            state["_bt_span"] = bt_rollout
            _bt._pending_rollout_span.set(None)

        async def rollout_loop() -> None:
            nonlocal state, bt_rollout
            state["timing"].generation.start = time.time()
            state["timing"].setup.start = time.time()
            bt_setup = _bt.setup_started(
                bt_rollout,
                env_id=_env_id,
                trajectory_id=state.get("trajectory_id", ""),
            )
            try:
                setup_state = await self.setup_state(state)
                if setup_state is not None:
                    state = setup_state
            except vf.Error as e:
                state["error"] = e
            finally:
                state["timing"].setup.end = time.time()
                _bt.setup_completed(
                    bt_setup,
                    duration_s=state["timing"].setup.end - state["timing"].setup.start,
                    error=repr(state["error"])[:500] if state.get("error") else "",
                )
            while not await self.is_completed(state):
                turn_t0 = time.monotonic()
                turn_idx = len(state["trajectory"])
                bt_turn = _bt.turn_started(
                    bt_rollout,
                    turn_index=turn_idx,
                    trajectory_id=state.get("trajectory_id", ""),
                )
                state["_bt_turn_span"] = bt_turn
                turn_err = ""
                env_dur: float | None = None
                model_dur: float | None = None
                try:
                    timing = state["timing"]
                    start_time = time.time()
                    prompt_messages = await self.get_prompt_messages(state)
                    end_time = time.time()
                    if state["trajectory"]:
                        timing.env.spans.append(
                            TimeSpan(start=start_time, end=end_time)
                        )
                        env_dur = end_time - start_time

                    prompt_messages = maybe_normalize_messages(
                        prompt_messages, field_name="prompt_messages"
                    )
                    if state.get("final_env_response") is not None:
                        continue

                    start_time = time.time()
                    response = await self.get_model_response(state, prompt_messages)
                    end_time = time.time()
                    model_dur = end_time - start_time
                    timing.model.spans.append(TimeSpan(start=start_time, end=end_time))
                    await self.add_model_response(state, prompt_messages, response)
                except vf.Error as e:
                    turn_err = repr(e)[:500]
                    if isinstance(e, vf.OverlongPromptError):
                        state["prompt_too_long"] = True
                        state["is_truncated"] = True
                    else:
                        state["error"] = e
                finally:
                    _bt.turn_completed(
                        bt_turn,
                        duration_s=time.monotonic() - turn_t0,
                        model_duration_s=model_dur,
                        env_duration_s=env_dur,
                        is_truncated=state.get("is_truncated", False),
                        error=turn_err,
                    )
                    state.pop("_bt_turn_span", None)

        try:
            await asyncio.wait_for(rollout_loop(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            self.mark_timed_out(state)
            _bt.timeout_triggered(bt_rollout, timeout_seconds=self.timeout_seconds)
        finally:
            await self._finalize_rollout(state)
        return state

    # MultiTurnEnv.rollout is decorated with @final, which is just a type hint
    # marker — it doesn't wrap the function. We replace the method directly.
    MultiTurnEnv.rollout = _traced_rollout  # type: ignore[assignment]
    _mark_patched(MultiTurnEnv)


# ---------------------------------------------------------------------------
# ToolEnv patches
# ---------------------------------------------------------------------------


def _patch_tool_env() -> None:
    from verifiers.envs.tool_env import ToolEnv

    if _is_patched(ToolEnv):
        return

    _orig_call_tool = ToolEnv.call_tool

    async def _traced_call_tool(
        self: Any,
        tool_name: str,
        tool_args: dict,
        tool_call_id: str,
        **kwargs: Any,
    ) -> Any:
        state = kwargs.get("state")
        bt_parent = (
            (state.get("_bt_turn_span") or state.get("_bt_span")) if state else None
        )
        bt_span = _bt.tool_call_started(
            bt_parent,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_args=tool_args,
        )
        t0 = time.monotonic()
        err_msg = ""
        result_content = None
        try:
            result = await _orig_call_tool(
                self, tool_name, tool_args, tool_call_id, **kwargs
            )
            result_content = (
                result.content if hasattr(result, "content") else str(result)
            )
            return result
        except Exception as exc:
            err_msg = repr(exc)[:500]
            raise
        finally:
            _bt.tool_call_completed(
                bt_span,
                duration_s=time.monotonic() - t0,
                result=result_content if not err_msg else None,
                error=err_msg,
            )

    # Also patch env_response to pass state= to call_tool
    async def _traced_env_response(
        self: Any,
        messages: Any,
        state: Any,
        **kwargs: Any,
    ) -> Any:
        import json
        from typing import cast

        import verifiers as vf
        from verifiers.types import ToolMessage

        last_msg = cast(vf.AssistantMessage, messages[-1])
        assert last_msg.tool_calls is not None
        tool_messages = []
        for tool_call in last_msg.tool_calls:
            tool_call_id: str = tool_call.id
            try:
                tool_name: str = tool_call.name
                tool_args: dict = json.loads(tool_call.arguments)
            except Exception as e:
                if self._should_stop_for_error(e):
                    raise vf.ToolParseError from e
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(e),
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            try:
                tool_message = await self.call_tool(
                    tool_name, tool_args, tool_call_id, state=state
                )
                tool_messages.append(tool_message)
            except Exception as e:
                if self._should_stop_for_error(e):
                    raise vf.ToolCallError from e
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(e),
                        tool_call_id=tool_call_id,
                    )
                )

        return tool_messages

    ToolEnv.call_tool = _traced_call_tool  # type: ignore[assignment]
    ToolEnv.env_response = _traced_env_response  # type: ignore[assignment]
    _mark_patched(ToolEnv)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_verifiers_tracing(
    *,
    api_key: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Activate Braintrust tracing for all verifiers environments.

    Call this once at the top of your environment file::

        from verifiers.envs.experimental.braintrust_tracing.integration import (
            setup_verifiers_tracing,
        )
        setup_verifiers_tracing()

    This patches ``Environment``, ``MultiTurnEnv``, and ``ToolEnv`` so that
    every rollout, model request, tool call, and scoring phase is traced as
    nested spans in Braintrust.  The patching is idempotent.

    Args:
        api_key: Braintrust API key.  Falls back to ``BRAINTRUST_API_KEY`` env var.
        project: Braintrust project name.  Falls back to ``VF_BRAINTRUST_PROJECT``
            env var, then ``"verifiers"``.
        tags: Optional run-level tags attached to every trace.  When ``None``
            a unique tag is auto-generated per ``generate()`` call.

    Returns:
        ``True`` if tracing is active (API key configured and ``braintrust``
        installed), ``False`` otherwise.
    """
    import os

    if api_key is not None:
        os.environ["BRAINTRUST_API_KEY"] = api_key
    if project is not None:
        os.environ["VF_BRAINTRUST_PROJECT"] = project

    # Force re-initialization of the tracing singleton if env vars were set
    if api_key is not None or project is not None:
        _bt._INSTANCE = None

    _patch_environment()
    _patch_multiturn_env()
    _patch_tool_env()

    if tags is not None:
        _bt.set_run_tags(tags)

    is_active = _bt.enabled()
    if is_active:
        _log.info("Braintrust tracing integration activated")
    else:
        _log.debug(
            "Braintrust tracing integration installed but inactive "
            "(set BRAINTRUST_API_KEY to enable)"
        )
    return is_active
