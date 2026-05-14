import importlib
import inspect
import time
from collections.abc import (
    Awaitable,
    Callable,
    Iterable,
    MutableSequence,
    Sequence,
)
from typing import Literal, cast

from typing_extensions import TypedDict

from verifiers.utils.async_utils import maybe_call_with_named_args

from .binding_utils import ROLLOUT_FRAMEWORK_ARGS, function_name
from .timing_utils import record_scoring_timing
from ..types import ConfigData, ConfigMap, GroupHandler, Handler

SignalKind = Literal["metric", "reward", "advantage"]
SignalStage = Literal["rollout", "group"]
SignalConfigMap = dict[str, ConfigMap]
SIGNAL_CONFIG_KEYS = {"stage", "priority", "weight", "skip"}


class SignalRecord(TypedDict):
    fn: Handler | GroupHandler
    name: str
    kind: SignalKind
    stage: SignalStage
    priority: int
    weight: float


def build_signals(
    owner: object | None = None,
    scoring: SignalConfigMap | None = None,
    metrics: Iterable[Handler] | None = None,
    rewards: Iterable[Handler] | None = None,
    advantages: Iterable[Handler] | None = None,
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


def add_metric(signals: MutableSequence[SignalRecord], fn: Handler) -> None:
    add_signal(signals, signal_from_function(fn, "metric"))


def add_reward(signals: MutableSequence[SignalRecord], fn: Handler) -> None:
    add_signal(signals, signal_from_function(fn, "reward"))


def add_advantage(signals: MutableSequence[SignalRecord], fn: Handler) -> None:
    add_signal(signals, signal_from_function(fn, "advantage"))


async def score_rollout(
    signals: Iterable[SignalRecord],
    task: ConfigMap,
    state: ConfigData,
    resolve_kwargs: Callable[
        [
            Handler,
            ConfigMap,
            ConfigData,
            set[str],
        ],
        Awaitable[ConfigData],
    ]
    | None = None,
) -> ConfigData:
    start_time = time.time()
    reward = float_value(state.get("reward"), 0.0)
    metrics = dict(cast(dict[str, float], state.get("metrics") or {}))
    framework_kwargs = rollout_framework_kwargs(task, state)
    protected_args = set(framework_kwargs)
    for signal in sorted(signals, key=signal_sort_key):
        if signal["stage"] != "rollout":
            continue
        extra_kwargs: ConfigData = {}
        if resolve_kwargs is not None:
            extra_kwargs = await resolve_kwargs(
                cast(Handler, signal["fn"]),
                task,
                state,
                protected_args,
            )
        value = await call_rollout_signal(signal, framework_kwargs, extra_kwargs)
        metrics[cast(str, signal["name"])] = value
        if signal["kind"] == "reward":
            reward += value * cast(float, signal["weight"])
    state["metrics"] = metrics
    state["reward"] = reward
    record_scoring_timing(state, start_time)
    return state


async def score_group(
    signals: Iterable[SignalRecord],
    tasks: list[ConfigMap],
    states: list[ConfigData],
    resolve_kwargs: Callable[
        [
            Handler,
            list[ConfigMap],
            list[ConfigData],
            set[str],
        ],
        Awaitable[ConfigData],
    ]
    | None = None,
) -> list[ConfigData]:
    start_time = time.time()
    rewards = [float_value(state.get("reward"), 0.0) for state in states]
    advantage_signals: list[SignalRecord] = []
    framework_kwargs = group_framework_kwargs(tasks, states)
    protected_args = set(framework_kwargs)
    for signal in sorted(signals, key=signal_sort_key):
        if signal["stage"] != "group":
            continue
        if signal["kind"] == "advantage":
            advantage_signals.append(signal)
            continue
        extra_kwargs: ConfigData = {}
        if resolve_kwargs is not None:
            extra_kwargs = await resolve_kwargs(
                cast(Handler, signal["fn"]),
                tasks,
                states,
                protected_args,
            )
        values = await call_group_signal(signal, framework_kwargs, extra_kwargs)
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
                cast(Handler, signal["fn"]),
                tasks,
                states,
                protected_args,
            )
        advantages = await call_group_signal(signal, framework_kwargs, extra_kwargs)
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


def signal_from_function(fn: Handler, kind: SignalKind | None = None) -> SignalRecord:
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


def apply_signal_config(signal: SignalRecord, config: ConfigMap) -> SignalRecord:
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


def decorated_kind(fn: Handler) -> SignalKind | None:
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
    fn = cast(Handler, signal["fn"])
    inspect.signature(fn)
    if signal["stage"] == "rollout":
        if signal["kind"] == "advantage":
            raise ValueError(
                f"Advantage signal {signal['name']!r} must use stage='group'."
            )


async def call_rollout_signal(
    signal: SignalRecord,
    framework_kwargs: ConfigMap,
    extra_kwargs: ConfigMap | None = None,
) -> float:
    fn = cast(GroupHandler, signal["fn"])
    kwargs = {**dict(extra_kwargs or {}), **dict(framework_kwargs)}
    validate_required_kwargs(fn, kwargs, signal_context(signal))
    value = await maybe_call_with_named_args(fn, **kwargs)
    return float(value)


async def call_group_signal(
    signal: SignalRecord,
    framework_kwargs: ConfigMap,
    extra_kwargs: ConfigMap | None = None,
) -> list[float]:
    fn = cast(Handler, signal["fn"])
    kwargs = {**dict(extra_kwargs or {}), **dict(framework_kwargs)}
    validate_required_kwargs(fn, kwargs, signal_context(signal))
    value = await maybe_call_with_named_args(fn, **kwargs)
    name = cast(str, signal["name"])
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"Group signal {name!r} must return a list of floats.")
    values = [float(item) for item in value]
    states = cast(list[ConfigData], framework_kwargs["states"])
    if len(values) != len(states):
        raise ValueError(
            f"Group signal {name!r} returned {len(values)} values for "
            f"{len(states)} states."
        )
    return values


def rollout_framework_kwargs(task: ConfigMap, state: ConfigData) -> ConfigData:
    kwargs: ConfigData = {"task": task, "state": state}
    for name in sorted(ROLLOUT_FRAMEWORK_ARGS - {"task", "state"}):
        if name in state:
            kwargs[name] = state[name]
        elif name in task:
            kwargs[name] = task[name]
    return kwargs


def group_framework_kwargs(
    tasks: list[ConfigMap], states: list[ConfigData]
) -> ConfigData:
    return {"tasks": tasks, "states": states}


def validate_required_kwargs(fn: Handler, kwargs: ConfigMap, context: str) -> None:
    signature = inspect.signature(fn)
    missing: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.default is not inspect.Parameter.empty:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.name not in kwargs:
            missing.append(parameter.name)
    if missing:
        raise TypeError(
            f"{context} has unresolved required args: {', '.join(missing)}."
        )


def signal_context(signal: SignalRecord) -> str:
    return f"{signal['kind']} signal {signal['name']!r}"


def import_ref(ref: str | None) -> Handler:
    if ref is None:
        raise ValueError("Import ref is required.")
    module_name, separator, attr_name = ref.partition(":")
    if not separator:
        raise ValueError(f"Signal ref {ref!r} must use 'module:object'.")
    obj = getattr(importlib.import_module(module_name), attr_name)
    if not callable(obj):
        raise TypeError(f"Signal ref {ref!r} did not resolve to a callable.")
    return cast(Handler, obj)


def validate_signal_config(name: str, config: ConfigMap) -> None:
    unknown_keys = set(config) - SIGNAL_CONFIG_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Signal config {name!r} has unknown keys: {unknown}.")


def get_optional_str(config: ConfigMap, key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Signal config key {key!r} must be a string.")
    return value


def get_optional_stage(config: ConfigMap) -> SignalStage | None:
    value = get_optional_str(config, "stage")
    if value is None:
        return None
    if value not in {"rollout", "group"}:
        raise ValueError("Signal stage must be 'rollout' or 'group'.")
    return cast(SignalStage, value)


def get_optional_number(config: ConfigMap, key: str) -> int | float | None:
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise TypeError(f"Signal config key {key!r} must be a number.")
    return value


def bool_config(config: ConfigMap, key: str, default: bool) -> bool:
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


def apply_advantage_to_trajectory(state: ConfigData, advantage: float) -> None:
    trajectory = state.get("trajectory", [])
    if not isinstance(trajectory, list):
        return
    for step in trajectory:
        if isinstance(step, dict):
            step = cast(ConfigData, step)
            if step.get("advantage") is None:
                step["advantage"] = advantage


def signal_sort_key(signal: SignalRecord) -> tuple[int, str, str, str]:
    return (
        -cast(int, signal["priority"]),
        cast(str, signal["name"]),
        cast(str, signal["kind"]),
        cast(str, signal["stage"]),
    )
