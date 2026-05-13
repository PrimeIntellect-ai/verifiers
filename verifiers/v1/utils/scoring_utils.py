from __future__ import annotations

import importlib
import inspect
import time
from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import Literal, cast

from verifiers.utils.async_utils import maybe_await

from .binding_utils import function_name
from .timing_utils import record_scoring_timing

SignalKind = Literal["metric", "reward", "advantage"]
SignalStage = Literal["rollout", "group"]
SignalRecord = dict[str, object]
SignalConfigMap = Mapping[str, Mapping[str, object]]
SIGNAL_CONFIG_KEYS = {"stage", "priority", "weight", "skip"}


def build_signals(
    owner: object | None = None,
    scoring: SignalConfigMap | None = None,
    metrics: Iterable[Callable[..., object]] | None = None,
    rewards: Iterable[Callable[..., object]] | None = None,
    advantages: Iterable[Callable[..., object]] | None = None,
) -> list[SignalRecord]:
    signals: list[SignalRecord] = []
    if owner is not None:
        for signal in decorated_signals(owner):
            add_signal(signals, signal)
    for fn in metrics or ():
        add_metric(signals, fn)
    for fn in rewards or ():
        add_reward(signals, fn)
    for fn in advantages or ():
        add_advantage(signals, fn)
    apply_scoring_config(signals, scoring or {})
    return sorted(signals, key=signal_sort_key)


def collect_signals(*signal_lists: Iterable[SignalRecord]) -> list[SignalRecord]:
    signals: list[SignalRecord] = []
    seen: set[str] = set()
    for signal_list in signal_lists:
        for signal in signal_list:
            name = cast(str, signal["name"])
            if name in seen:
                raise ValueError(f"Signal {name!r} is defined twice.")
            seen.add(name)
            signals.append(signal)
    return sorted(signals, key=signal_sort_key)


def add_metric(
    signals: MutableSequence[SignalRecord], fn: Callable[..., object]
) -> None:
    add_signal(signals, signal_from_function(fn, "metric"))


def add_reward(
    signals: MutableSequence[SignalRecord], fn: Callable[..., object]
) -> None:
    add_signal(signals, signal_from_function(fn, "reward"))


def add_advantage(
    signals: MutableSequence[SignalRecord], fn: Callable[..., object]
) -> None:
    add_signal(signals, signal_from_function(fn, "advantage"))


async def score_rollout(
    signals: Iterable[SignalRecord],
    task: Mapping[str, object],
    state: dict[str, object],
    resolve_kwargs: Callable[
        [Callable[..., object], Mapping[str, object], dict[str, object]],
        Awaitable[dict[str, object]],
    ]
    | None = None,
) -> dict[str, object]:
    start_time = time.time()
    reward = float_value(state.get("reward"), 0.0)
    metrics = dict(cast(dict[str, float], state.get("metrics") or {}))
    for signal in sorted(signals, key=signal_sort_key):
        if signal["stage"] != "rollout":
            continue
        extra_kwargs: dict[str, object] = {}
        if resolve_kwargs is not None:
            extra_kwargs = await resolve_kwargs(
                cast(Callable[..., object], signal["fn"]), task, state
            )
        value = await call_rollout_signal(signal, task, state, extra_kwargs)
        metrics[cast(str, signal["name"])] = value
        if signal["kind"] == "reward":
            reward += value * cast(float, signal["weight"])
    state["metrics"] = metrics
    state["reward"] = reward
    record_scoring_timing(state, start_time)
    return state


async def score_group(
    signals: Iterable[SignalRecord],
    tasks: list[Mapping[str, object]],
    states: list[dict[str, object]],
    resolve_kwargs: Callable[
        [Callable[..., object], list[Mapping[str, object]], list[dict[str, object]]],
        Awaitable[dict[str, object]],
    ]
    | None = None,
) -> list[dict[str, object]]:
    start_time = time.time()
    rewards = [float_value(state.get("reward"), 0.0) for state in states]
    advantage_signals: list[SignalRecord] = []
    for signal in sorted(signals, key=signal_sort_key):
        if signal["stage"] != "group":
            continue
        if signal["kind"] == "advantage":
            advantage_signals.append(signal)
            continue
        extra_kwargs: dict[str, object] = {}
        if resolve_kwargs is not None:
            extra_kwargs = await resolve_kwargs(
                cast(Callable[..., object], signal["fn"]), tasks, states
            )
        values = await call_group_signal(signal, tasks, states, extra_kwargs)
        for index, value in enumerate(values):
            metrics = dict(cast(dict[str, float], states[index].get("metrics") or {}))
            metrics[cast(str, signal["name"])] = value
            states[index]["metrics"] = metrics
            if signal["kind"] == "reward":
                rewards[index] += value * cast(float, signal["weight"])
    advantages: list[float] | None = None
    for signal in advantage_signals:
        extra_kwargs = {}
        if resolve_kwargs is not None:
            extra_kwargs = await resolve_kwargs(
                cast(Callable[..., object], signal["fn"]), tasks, states
            )
        advantages = await call_group_signal(signal, tasks, states, extra_kwargs)
    for index, state in enumerate(states):
        state["reward"] = rewards[index]
        if advantages is not None:
            state["advantage"] = advantages[index]
            apply_advantage_to_trajectory(state, advantages[index])
        record_scoring_timing(state, start_time)
    return states


def add_signal(signals: MutableSequence[SignalRecord], signal: SignalRecord) -> None:
    name = cast(str, signal["name"])
    if any(existing["name"] == name for existing in signals):
        raise ValueError(f"Signal {name!r} is defined twice.")
    validate_signal(signal)
    signals.append(signal)


def apply_scoring_config(
    signals: MutableSequence[SignalRecord], scoring: SignalConfigMap
) -> None:
    by_name = {cast(str, signal["name"]): signal for signal in signals}
    for name, config in scoring.items():
        validate_signal_config(name, config)
        if bool_config(config, "skip", default=False):
            if name not in by_name:
                raise ValueError(f"Cannot skip unknown signal {name!r}.")
            signals.remove(by_name[name])
            del by_name[name]
            continue
        if name not in by_name:
            raise ValueError(f"Config references unknown signal {name!r}.")
        signal = apply_signal_config(by_name[name], config)
        validate_signal(signal)
        index = signals.index(by_name[name])
        signals[index] = signal
        by_name[name] = signal


def decorated_signals(owner: object) -> list[SignalRecord]:
    signals: list[SignalRecord] = []
    for _, method in inspect.getmembers(owner, predicate=callable):
        if getattr(method, "metric", False):
            signals.append(signal_from_function(method, "metric"))
        if getattr(method, "reward", False):
            signals.append(signal_from_function(method, "reward"))
        if getattr(method, "advantage", False):
            signals.append(signal_from_function(method, "advantage"))
    return signals


def signal_from_function(
    fn: Callable[..., object], kind: SignalKind | None = None
) -> SignalRecord:
    inferred_kind = decorated_kind(fn)
    if kind is not None and inferred_kind is not None and kind != inferred_kind:
        raise ValueError(
            f"Signal function {function_name(fn)!r} is decorated as {inferred_kind!r}."
        )
    resolved_kind = kind or inferred_kind
    if resolved_kind is None:
        raise ValueError(
            f"Signal function {function_name(fn)!r} must be decorated or given a kind."
        )
    priority = int(getattr(fn, f"{resolved_kind}_priority", 0))
    stage = cast(SignalStage, getattr(fn, f"{resolved_kind}_stage", "rollout"))
    weight = 0.0
    if resolved_kind == "reward":
        weight = float(getattr(fn, "reward_weight", 1.0))
    return {
        "fn": fn,
        "name": function_name(fn),
        "kind": resolved_kind,
        "stage": stage,
        "priority": priority,
        "weight": weight,
    }


def apply_signal_config(
    signal: SignalRecord, config: Mapping[str, object]
) -> SignalRecord:
    kind = cast(SignalKind, signal["kind"])
    stage = get_optional_stage(config) or cast(SignalStage, signal["stage"])
    priority_value = get_optional_number(config, "priority")
    priority = cast(int, signal["priority"])
    if priority_value is not None:
        priority = int(priority_value)
    weight = cast(float, signal["weight"])
    if kind in {"metric", "advantage"}:
        weight = 0.0
    else:
        weight_value = get_optional_number(config, "weight")
        if weight_value is not None:
            weight = float(weight_value)
    return {
        "fn": signal["fn"],
        "name": signal["name"],
        "kind": kind,
        "stage": stage,
        "priority": priority,
        "weight": weight,
    }


def decorated_kind(fn: Callable[..., object]) -> SignalKind | None:
    has_metric = bool(getattr(fn, "metric", False))
    has_reward = bool(getattr(fn, "reward", False))
    has_advantage = bool(getattr(fn, "advantage", False))
    if sum([has_metric, has_reward, has_advantage]) > 1:
        raise ValueError(f"Signal function {function_name(fn)!r} has two kinds.")
    if has_metric:
        return "metric"
    if has_reward:
        return "reward"
    if has_advantage:
        return "advantage"
    return None


def validate_signal(signal: SignalRecord) -> None:
    fn = cast(Callable[..., object], signal["fn"])
    names = set(inspect.signature(fn).parameters)
    if signal["stage"] == "rollout":
        if signal["kind"] == "advantage":
            raise ValueError(
                f"Advantage signal {signal['name']!r} must use stage='group'."
            )
        if not {"task", "state"}.issubset(names):
            raise ValueError(
                f"Rollout signal {signal['name']!r} must accept task and state."
            )
    if signal["stage"] == "group":
        if not {"tasks", "states"}.issubset(names):
            raise ValueError(
                f"Group signal {signal['name']!r} must accept tasks and states."
            )


async def call_rollout_signal(
    signal: SignalRecord,
    task: Mapping[str, object],
    state: dict[str, object],
    extra_kwargs: Mapping[str, object] | None = None,
) -> float:
    value = await maybe_await(
        cast(Callable[..., object], signal["fn"]),
        task=task,
        state=state,
        **dict(extra_kwargs or {}),
    )
    return float(value)


async def call_group_signal(
    signal: SignalRecord,
    tasks: list[Mapping[str, object]],
    states: list[dict[str, object]],
    extra_kwargs: Mapping[str, object] | None = None,
) -> list[float]:
    value = await maybe_await(
        cast(Callable[..., object], signal["fn"]),
        tasks=tasks,
        states=states,
        **dict(extra_kwargs or {}),
    )
    name = cast(str, signal["name"])
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"Group signal {name!r} must return a list of floats.")
    values = [float(item) for item in value]
    if len(values) != len(states):
        raise ValueError(
            f"Group signal {name!r} returned {len(values)} values for "
            f"{len(states)} states."
        )
    return values


def import_ref(ref: str | None) -> Callable[..., object]:
    if ref is None:
        raise ValueError("Import ref is required.")
    module_name, separator, attr_name = ref.partition(":")
    if not separator:
        raise ValueError(f"Signal ref {ref!r} must use 'module:object'.")
    obj = getattr(importlib.import_module(module_name), attr_name)
    if not callable(obj):
        raise TypeError(f"Signal ref {ref!r} did not resolve to a callable.")
    return cast(Callable[..., object], obj)


def validate_signal_config(name: str, config: Mapping[str, object]) -> None:
    unknown_keys = set(config) - SIGNAL_CONFIG_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Signal config {name!r} has unknown keys: {unknown}.")


def get_optional_str(config: Mapping[str, object], key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Signal config key {key!r} must be a string.")
    return value


def get_optional_stage(config: Mapping[str, object]) -> SignalStage | None:
    value = get_optional_str(config, "stage")
    if value is None:
        return None
    if value not in {"rollout", "group"}:
        raise ValueError("Signal stage must be 'rollout' or 'group'.")
    return cast(SignalStage, value)


def get_optional_number(config: Mapping[str, object], key: str) -> int | float | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise TypeError(f"Signal config key {key!r} must be a number.")
    return value


def bool_config(config: Mapping[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"Signal config key {key!r} must be a boolean.")
    return value


def float_value(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return default
    return float(value or 0.0)


def apply_advantage_to_trajectory(state: dict[str, object], advantage: float) -> None:
    trajectory = state.get("trajectory", [])
    if not isinstance(trajectory, list):
        return
    for step in trajectory:
        if isinstance(step, dict):
            step = cast(dict[str, object], step)
            if step.get("advantage") is None:
                step["advantage"] = advantage


def signal_sort_key(signal: SignalRecord) -> tuple[int, str, str, str]:
    return (
        -cast(int, signal["priority"]),
        cast(str, signal["name"]),
        cast(str, signal["kind"]),
        cast(str, signal["stage"]),
    )
