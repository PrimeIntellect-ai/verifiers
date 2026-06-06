from __future__ import annotations

import math

import verifiers.v1 as vf


@vf.advantage
def grpo(tasks: list[vf.Task], states: list[vf.State]) -> None:
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
        state.advantage = float(value)
        for turn in state.transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [0.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    float(value) for _ in turn.tokens.completion_ids
                ]


@vf.advantage
def rloo(tasks: list[vf.Task], states: list[vf.State]) -> None:
    _ = tasks
    if not states:
        return
    if len(states) == 1:
        states[0].advantage = 0.0
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
        state.advantage = float(value)
        for turn in state.transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [0.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    float(value) for _ in turn.tokens.completion_ids
                ]


@vf.advantage
def reinforce(tasks: list[vf.Task], states: list[vf.State]) -> None:
    _ = tasks
    for state in states:
        value = float(state.reward)
        state.advantage = value
        for turn in state.transcript:
            if turn.tokens is not None:
                turn.tokens.prompt_advantages = [0.0 for _ in turn.tokens.prompt_ids]
                turn.tokens.completion_advantages = [
                    value for _ in turn.tokens.completion_ids
                ]
