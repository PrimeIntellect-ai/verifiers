from __future__ import annotations

import math
from typing import TypeAlias, cast

from .decorators import advantage
from .state import State
from .task import Task
from .types import Handler
from .utils.scoring_utils import SignalRecord, signal_from_function
from .utils.config_utils import import_config_ref


AdvantageConfig: TypeAlias = str | None


def resolve_config(value: AdvantageConfig) -> Handler | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError("Env advantage must be a non-empty function path.")
    ref = value if ":" in value else f"verifiers.v1.advantages:{value}"
    resolved = import_config_ref(ref)
    if not callable(resolved):
        raise TypeError(f"Env advantage {ref!r} must resolve to a callable.")
    if not getattr(resolved, "advantage", False):
        raise TypeError(f"Env advantage {ref!r} must be decorated with @vf.advantage.")
    return cast(Handler, resolved)


def signal(fn: Handler) -> SignalRecord:
    return signal_from_function(fn)


@advantage
def rl(tasks: list[Task], states: list[State]) -> None:
    grpo(tasks, states)


@advantage
def grpo(tasks: list[Task], states: list[State]) -> None:
    _ = tasks
    if not states:
        return
    baseline = sum(state.reward for state in states) / len(states)
    values = [float(state.reward - baseline) for state in states]
    variance = sum(value * value for value in values) / len(values)
    scale = math.sqrt(variance)
    if scale == 0.0:
        values = [0.0 for _ in values]
    else:
        values = [float(value / scale) for value in values]
    for state, value in zip(states, values, strict=True):
        for turn in state.transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [0.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    float(value) for _ in turn.tokens.completion_ids
                ]


@advantage
def rloo(tasks: list[Task], states: list[State]) -> None:
    _ = tasks
    if not states:
        return
    if len(states) == 1:
        for turn in states[0].transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [0.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    0.0 for _ in turn.tokens.completion_ids
                ]
        return
    total = sum(state.reward for state in states)
    values = [
        float(state.reward - ((total - state.reward) / (len(states) - 1)))
        for state in states
    ]
    for state, value in zip(states, values, strict=True):
        for turn in state.transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [0.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    float(value) for _ in turn.tokens.completion_ids
                ]


@advantage
def reinforce(tasks: list[Task], states: list[State]) -> None:
    _ = tasks
    for state in states:
        value = float(state.reward)
        for turn in state.transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [0.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    value for _ in turn.tokens.completion_ids
                ]


@advantage
def sft(tasks: list[Task], states: list[State]) -> None:
    _ = tasks
    for state in states:
        for turn in state.transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [1.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    1.0 for _ in turn.tokens.completion_ids
                ]
