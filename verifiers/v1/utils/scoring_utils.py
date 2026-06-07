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

from verifiers.types import Messages
from verifiers.utils.async_utils import maybe_call_with_named_args

from ..runtime import Runtime
from ..state import State, Turn
from ..task import Task
from ..types import Context, Handler, JsonData, ModelClient

SignalKind = Literal["metric", "reward", "advantage"]
SignalStage = Literal["rollout", "group"]


class SignalRecord(TypedDict):
    fn: Handler
    name: str
    kind: SignalKind
    stage: SignalStage
    priority: int
    weight: float


SignalKwarg = (
    Task
    | State
    | list[Task]
    | list[State]
    | list[Turn]
    | Messages
    | JsonData
    | dict[str, float]
    | float
    | int
    | str
    | ModelClient
    | Runtime
    | Context
    | None
)
SignalKwargs = dict[str, SignalKwarg]


def build_signals(
    owner: object | None = None,
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
    return sorted(signals, key=signal_sort_key)


def collect_signals(*signal_lists: Iterable[SignalRecord]) -> list[SignalRecord]:
    signals: list[SignalRecord] = []
    seen: set[str] = set()
    for signal_list in signal_lists:
        for signal in signal_list:
            name = signal["name"]
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
    task: Task,
    state: State,
    runtime: Runtime | None = None,
    model_client: ModelClient | None = None,
    teacher: ModelClient | None = None,
    context: Context | None = None,
    resolve_kwargs: Callable[
        [
            Handler,
            Task,
            State,
            set[str],
        ],
        Awaitable[SignalKwargs],
    ]
    | None = None,
) -> State:
    start_time = time.time()
    reward = float(state.reward)
    metrics = dict(state.metrics)
    framework_kwargs = rollout_framework_kwargs(
        task,
        state,
        runtime=runtime,
        model_client=model_client,
        teacher=teacher,
        context=context,
    )
    protected_args = set(framework_kwargs)
    for signal in sorted(signals, key=signal_sort_key):
        if signal["stage"] != "rollout":
            continue
        extra_kwargs: SignalKwargs = {}
        if resolve_kwargs is not None:
            extra_kwargs = await resolve_kwargs(
                signal["fn"],
                task,
                state,
                protected_args,
            )
        value = await call_rollout_signal(signal, framework_kwargs, extra_kwargs)
        metrics[signal["name"]] = value
        if signal["kind"] == "reward":
            reward += value * signal["weight"]
    state.metrics = metrics
    state.reward = reward
    state.timing.scoring.start = start_time
    state.timing.scoring.end = time.time()
    return state


async def score_group(
    signals: Iterable[SignalRecord],
    tasks: list[Task],
    states: list[State],
    model_client: ModelClient | None = None,
    teacher: ModelClient | None = None,
    resolve_kwargs: Callable[
        [
            Handler,
            list[Task],
            list[State],
            set[str],
        ],
        Awaitable[SignalKwargs],
    ]
    | None = None,
) -> list[State]:
    start_time = time.time()
    rewards = [float(state.reward) for state in states]
    advantage_signals: list[SignalRecord] = []
    framework_kwargs = group_framework_kwargs(
        tasks, states, model_client=model_client, teacher=teacher
    )
    protected_args = set(framework_kwargs)
    for signal in sorted(signals, key=signal_sort_key):
        if signal["stage"] != "group":
            continue
        if signal["kind"] == "advantage":
            advantage_signals.append(signal)
            continue
        extra_kwargs: SignalKwargs = {}
        if resolve_kwargs is not None:
            extra_kwargs = await resolve_kwargs(
                signal["fn"],
                tasks,
                states,
                protected_args,
            )
        values = await call_group_signal(signal, framework_kwargs, extra_kwargs)
        for index, value in enumerate(values):
            metrics = dict(states[index].metrics)
            metrics[signal["name"]] = value
            states[index].metrics = metrics
            if signal["kind"] == "reward":
                rewards[index] += value * signal["weight"]
        for index, state in enumerate(states):
            state.reward = rewards[index]
    for signal in advantage_signals:
        extra_kwargs: SignalKwargs = {}
        if resolve_kwargs is not None:
            extra_kwargs = await resolve_kwargs(
                signal["fn"],
                tasks,
                states,
                protected_args,
            )
        await call_group_advantage_signal(signal, framework_kwargs, extra_kwargs)
    for index, state in enumerate(states):
        state.reward = rewards[index]
        state.timing.scoring.start = start_time
        state.timing.scoring.end = time.time()
    return states


def add_signal(signals: MutableSequence[SignalRecord], signal: SignalRecord) -> None:
    name = signal["name"]
    if any(existing["name"] == name for existing in signals):
        raise ValueError(f"Signal {name!r} is defined twice.")
    validate_signal(signal)
    signals.append(signal)


def decorated_signals(owner: object) -> list[SignalRecord]:
    signals: list[SignalRecord] = []
    for _, method in inspect.getmembers(owner, predicate=callable):
        signal = cast(Handler, method)
        if getattr(method, "metric", False):
            signals.append(signal_from_function(signal, "metric"))
        if getattr(method, "reward", False):
            signals.append(signal_from_function(signal, "reward"))
        if getattr(method, "advantage", False):
            signals.append(signal_from_function(signal, "advantage"))
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
    raw_stage = getattr(fn, f"{resolved_kind}_stage", "rollout")
    if raw_stage not in ("rollout", "group"):
        raise ValueError(
            f"Signal function {function_name(fn)!r} has invalid stage {raw_stage!r}."
        )
    stage: SignalStage = raw_stage
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
    inspect.signature(signal["fn"])
    if signal["stage"] == "rollout":
        if signal["kind"] == "advantage":
            raise ValueError(
                f"Advantage signal {signal['name']!r} must use stage='group'."
            )


async def call_rollout_signal(
    signal: SignalRecord,
    framework_kwargs: SignalKwargs,
    extra_kwargs: SignalKwargs | None = None,
) -> float:
    fn = signal["fn"]
    kwargs = {**dict(extra_kwargs or {}), **dict(framework_kwargs)}
    validate_required_kwargs(fn, kwargs, signal_context(signal))
    value = await maybe_call_with_named_args(fn, **kwargs)
    return float(value)


async def call_group_signal(
    signal: SignalRecord,
    framework_kwargs: SignalKwargs,
    extra_kwargs: SignalKwargs | None = None,
) -> list[float]:
    fn = signal["fn"]
    kwargs = {**dict(extra_kwargs or {}), **dict(framework_kwargs)}
    validate_required_kwargs(fn, kwargs, signal_context(signal))
    value = await maybe_call_with_named_args(fn, **kwargs)
    name = signal["name"]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"Group signal {name!r} must return a list of floats.")
    values = [float(item) for item in value]
    states_value = framework_kwargs["states"]
    if not isinstance(states_value, list) or not all(
        isinstance(state, State) for state in states_value
    ):
        raise TypeError("Group signal framework kwargs must include states.")
    states = states_value
    if len(values) != len(states):
        raise ValueError(
            f"Group signal {name!r} returned {len(values)} values for "
            f"{len(states)} states."
        )
    return values


async def call_group_advantage_signal(
    signal: SignalRecord,
    framework_kwargs: SignalKwargs,
    extra_kwargs: SignalKwargs | None = None,
) -> None:
    fn = signal["fn"]
    kwargs = {**dict(extra_kwargs or {}), **dict(framework_kwargs)}
    validate_required_kwargs(fn, kwargs, signal_context(signal))
    value = await maybe_call_with_named_args(fn, **kwargs)
    if value is not None:
        raise TypeError(
            f"Group advantage signal {signal['name']!r} must mutate states in "
            f"place and return None, not {type(value).__name__}."
        )


def rollout_framework_kwargs(
    task: Task,
    state: State,
    *,
    runtime: Runtime | None = None,
    model_client: ModelClient | None = None,
    teacher: ModelClient | None = None,
    context: Context | None = None,
) -> SignalKwargs:
    kwargs: SignalKwargs = {
        "task": task,
        "state": state,
        "extras": state.extras,
        "transcript": state.transcript,
        "completion": state.completion,
        "metrics": state.metrics,
        "reward": state.reward,
        "prompt": state.prompt or task.prompt,
        "example_id": task.row_id,
        "model": model_client,
        "model_name": model_client.config.model if model_client is not None else None,
        "teacher": teacher,
        "teacher_name": teacher.config.model if teacher is not None else None,
        "context": context,
    }
    if runtime is not None:
        kwargs["runtime"] = runtime
    return kwargs


def group_framework_kwargs(
    tasks: list[Task],
    states: list[State],
    *,
    model_client: ModelClient | None = None,
    teacher: ModelClient | None = None,
) -> SignalKwargs:
    return {
        "tasks": tasks,
        "states": states,
        "model": model_client,
        "model_name": model_client.config.model if model_client is not None else None,
        "teacher": teacher,
        "teacher_name": teacher.config.model if teacher is not None else None,
    }


def validate_required_kwargs(fn: Handler, kwargs: SignalKwargs, context: str) -> None:
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


def signal_sort_key(signal: SignalRecord) -> tuple[int, str, str, str]:
    return (
        -signal["priority"],
        signal["name"],
        signal["kind"],
        signal["stage"],
    )


def function_name(fn: Handler) -> str:
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name:
        return name
    return type(fn).__name__
