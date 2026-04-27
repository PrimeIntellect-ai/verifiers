from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from verifiers.rubrics.rubric import Rubric
from verifiers.rubrics.rubric_group import RubricGroup
from verifiers.types import GroupRewardFunc, RewardFunc, State
from verifiers.utils.async_utils import maybe_call_with_named_args

from verifiers.envs.experimental.task import Task

SignalKind = Literal["metric", "reward", "advantage"]
SignalStage = Literal["rollout", "group"]


@dataclass(frozen=True)
class Signal:
    fn: Callable[..., object]
    name: str
    kind: SignalKind
    stage: SignalStage
    priority: int = 0
    weight: float = 0.0


class Scoring:
    """Executes rollout and group metric/reward signals."""

    def __init__(self, signals: Iterable[Signal] = ()):
        self.signals = sorted_signals(signals)

    @property
    def parser(self):
        return None

    @property
    def rollout_metrics(self) -> list[Signal]:
        return [
            signal
            for signal in self.signals
            if signal.stage == "rollout" and signal.kind == "metric"
        ]

    @property
    def rollout_rewards(self) -> list[Signal]:
        return [
            signal
            for signal in self.signals
            if signal.stage == "rollout" and signal.kind == "reward"
        ]

    @property
    def group_metrics(self) -> list[Signal]:
        return [
            signal
            for signal in self.signals
            if signal.stage == "group" and signal.kind == "metric"
        ]

    @property
    def group_rewards(self) -> list[Signal]:
        return [
            signal
            for signal in self.signals
            if signal.stage == "group" and signal.kind == "reward"
        ]

    @property
    def advantages(self) -> list[Signal]:
        return [signal for signal in self.signals if signal.kind == "advantage"]

    def signal_names(self) -> list[str]:
        return [signal.name for signal in self.signals if signal.kind != "advantage"]

    def has_group_signals(self) -> bool:
        return any(signal.stage == "group" for signal in self.signals)

    async def rollout(self, task: Task, state: State, resources: object) -> State:
        start_time = time.time()
        metrics = dict(cast(dict[str, float], state.get("metrics") or {}))
        reward = 0.0
        for signal in [*self.rollout_metrics, *self.rollout_rewards]:
            value = await call_rollout_signal(signal, task, state, resources)
            metrics[signal.name] = value
            if signal.kind == "reward":
                reward += value * signal.weight
        state["metrics"] = metrics
        state["reward"] = reward
        record_scoring_timing(state, start_time)
        return state

    async def group(
        self, tasks: list[Task], states: list[State], resources: object
    ) -> list[State]:
        start_time = time.time()
        rewards = [float(state.get("reward", 0.0) or 0.0) for state in states]
        for signal in self.group_metrics:
            values = await call_group_signal(signal, tasks, states, resources)
            apply_group_metric(signal.name, values, states)
        for signal in self.group_rewards:
            values = await call_group_signal(signal, tasks, states, resources)
            apply_group_metric(signal.name, values, states)
            for index, value in enumerate(values):
                rewards[index] += value * signal.weight
        for index, state in enumerate(states):
            state["reward"] = rewards[index]
        advantages = await self.compute_advantages(tasks, states, resources)
        apply_advantages(states, advantages)
        for state in states:
            record_scoring_timing(state, start_time)
        return states

    async def compute_advantages(
        self, tasks: list[Task], states: list[State], resources: object
    ) -> list[float]:
        if not states:
            return []
        if self.advantages:
            values: list[float] = [0.0] * len(states)
            for signal in self.advantages:
                values = await call_group_signal(signal, tasks, states, resources)
            return values
        rewards = [float(state.get("reward", 0.0) or 0.0) for state in states]
        avg_reward = sum(rewards) / len(rewards)
        return [reward - avg_reward for reward in rewards]


def sorted_signals(signals: Iterable[Signal]) -> list[Signal]:
    return sorted(
        signals,
        key=lambda signal: (-signal.priority, signal.name, signal.kind, signal.stage),
    )


async def call_rollout_signal(
    signal: Signal, task: Task, state: State, resources: object
) -> float:
    value = await maybe_call_with_named_args(
        signal.fn,
        task=task,
        state=state,
        resources=resources,
    )
    return float(value)


async def call_group_signal(
    signal: Signal, tasks: list[Task], states: list[State], resources: object
) -> list[float]:
    value = await maybe_call_with_named_args(
        signal.fn,
        tasks=tasks,
        states=states,
        resources=resources,
    )
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"Group signal {signal.name!r} must return a list of floats.")
    values = [float(item) for item in value]
    if len(values) != len(states):
        raise ValueError(
            f"Group signal {signal.name!r} returned {len(values)} scores for "
            f"{len(states)} states."
        )
    return values


def apply_group_metric(name: str, values: list[float], states: list[State]) -> None:
    for state, value in zip(states, values):
        metrics = dict(cast(dict[str, float], state.get("metrics") or {}))
        metrics[name] = value
        state["metrics"] = metrics


def apply_advantages(states: list[State], advantages: list[float]) -> None:
    for state, advantage in zip(states, advantages):
        state["advantage"] = advantage
        for step in state.get("trajectory", []):
            if step["advantage"] is None:
                step["advantage"] = advantage
            if step["reward"] is None:
                step["reward"] = state.get("reward")


def record_scoring_timing(state: State, start_time: float) -> None:
    timing = state.get("timing")
    if not isinstance(timing, dict):
        return
    scoring_ms = (time.time() - start_time) * 1000
    timing["scoring_ms"] = float(timing.get("scoring_ms", 0.0)) + scoring_ms
    timing["total_ms"] = float(timing.get("total_ms", 0.0)) + scoring_ms


def signals_from_configs(
    configs: list[object], context: object, kind: SignalKind
) -> list[Signal]:
    signals: list[Signal] = []
    for config in configs:
        for entry in signal_entries(config, kind):
            signals.append(signal_from_entry(entry, context, kind, strict=True))
    return signals


def signal_entries(config: object, kind: SignalKind) -> list[dict[str, object]]:
    if config is None:
        return []
    if callable(config) or isinstance(config, str):
        return [{"fn": config}]
    if isinstance(config, Sequence) and not isinstance(config, str | bytes):
        return [
            entry
            for item in config
            for entry in signal_entries(cast(object, item), kind)
        ]
    if isinstance(config, Mapping):
        mapping = cast(Mapping[str, object], config)
        plural = f"{kind}s"
        if "fn" in mapping:
            return [dict(mapping)]
        if plural in mapping:
            return signal_entries(mapping[plural], kind)
    raise TypeError(f"Unsupported {kind} channel config: {config!r}")


def signal_from_entry(
    entry: Mapping[str, object], context: object, kind: SignalKind, *, strict: bool
) -> Signal:
    fn = resolve_signal_function(entry["fn"], context)
    default_stage = "group" if kind == "advantage" else "rollout"
    stage = cast(
        SignalStage,
        entry.get("stage") or getattr(fn, f"{kind}_stage", default_stage),
    )
    name = signal_name(entry, fn)
    priority_value = entry.get("priority", getattr(fn, f"{kind}_priority", 0))
    priority = int(cast(int | str, priority_value))
    if kind == "metric":
        weight = 0.0
    elif kind == "advantage":
        weight = 0.0
    else:
        weight_value = entry.get("weight", getattr(fn, "reward_weight", 1.0))
        weight = float(cast(int | float | str, weight_value))
    signal = Signal(
        fn=fn,
        name=name,
        kind=kind,
        stage=stage,
        priority=priority,
        weight=weight,
    )
    validate_signal(signal, strict=strict)
    return signal


def resolve_signal_function(ref: object, context: object) -> Callable[..., object]:
    get_object = getattr(context, "get_object", None)
    obj = get_object(ref) if callable(get_object) else ref
    if not callable(obj):
        raise TypeError(f"Signal function {ref!r} did not resolve to a function.")
    return cast(Callable[..., object], obj)


def signal_name(entry: Mapping[str, object], fn: Callable[..., object]) -> str:
    raw_name = entry.get("name") or getattr(fn, "__name__", None)
    if not isinstance(raw_name, str) or not raw_name:
        raise ValueError("Signal entries require a stable name.")
    return raw_name


def validate_signal(signal: Signal, *, strict: bool) -> None:
    parameters = inspect.signature(signal.fn).parameters
    names = set(parameters)
    has_group_arg = bool(names & {"tasks", "states"})
    has_rollout_arg = bool(names & {"task", "state"})
    if signal.stage == "group":
        if strict and not has_group_arg:
            raise ValueError(
                f"Group signal {signal.name!r} must accept 'tasks' or 'states'."
            )
        if strict and has_rollout_arg:
            raise ValueError(
                f"Group signal {signal.name!r} must not accept singular task/state args."
            )
    else:
        if has_group_arg:
            raise ValueError(
                f"Rollout signal {signal.name!r} must not accept plural tasks/states "
                "args. Set stage='group' explicitly for group signals."
            )


def signals_from_rubric(rubric: Rubric) -> list[Signal]:
    signals: list[Signal] = []
    for child in flatten_rubrics(rubric):
        for func, weight in zip(child.funcs, child.weights):
            is_group = child._is_group_func(func)
            kind: SignalKind = "metric" if float(weight) == 0.0 else "reward"
            stage: SignalStage = "group" if is_group else "rollout"
            name = getattr(func, "__name__", repr(func))
            if is_group:
                wrapped = group_rubric_signal(child, func, name)
            else:
                wrapped = rollout_rubric_signal(child, func, name)
            signals.append(
                Signal(
                    fn=wrapped,
                    name=name,
                    kind=kind,
                    stage=stage,
                    priority=int(getattr(func, "reward_priority", 0)),
                    weight=float(weight),
                )
            )
    return sorted_signals(signals)


def rollout_rubric_signal(
    rubric: Rubric, func: Callable[..., object], name: str
) -> Callable[..., object]:
    async def wrapped(state: State):
        return await rubric._call_individual_reward_func(cast(RewardFunc, func), state)

    wrapped.__name__ = name
    return wrapped


def group_rubric_signal(
    rubric: Rubric, func: Callable[..., object], name: str
) -> Callable[..., object]:
    async def wrapped(states: list[State]):
        return await rubric._call_group_reward_func(cast(GroupRewardFunc, func), states)

    wrapped.__name__ = name
    return wrapped


def flatten_rubrics(rubric: Rubric) -> list[Rubric]:
    if isinstance(rubric, RubricGroup):
        flattened: list[Rubric] = []
        for child in rubric.rubrics:
            flattened.extend(flatten_rubrics(child))
        return flattened
    return [rubric]


def rubric_class_objects(rubric: Rubric) -> dict[str, object]:
    objects: dict[str, object] = {}
    for child in flatten_rubrics(rubric):
        for name, value in child.class_objects.items():
            if name == "resources":
                continue
            existing = objects.get(name)
            if existing is not None and existing is not value:
                raise ValueError(f"Rubric class object {name!r} is defined twice.")
            objects[name] = value
    return objects


def rubric_cleanup_handlers(
    rubric: Rubric, stage: SignalStage = "rollout"
) -> list[Callable[..., object]]:
    handlers: list[Callable[..., object]] = []
    for child in flatten_rubrics(rubric):
        handlers.extend(
            handler
            for handler in child._cleanup_handlers
            if getattr(handler, "cleanup_stage", "rollout") == stage
        )
    return handlers
