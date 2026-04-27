from __future__ import annotations

from collections.abc import Callable, Iterable

from verifiers.types import State
from verifiers.utils.async_utils import maybe_call_with_named_args

from verifiers.envs.experimental.task import Task

LifecycleHandler = Callable[..., object]


class Lifecycle:
    """Executes staged render, cleanup, and teardown handlers."""

    def __init__(
        self,
        *,
        render_rollout: Iterable[LifecycleHandler] = (),
        render_group: Iterable[LifecycleHandler] = (),
        cleanup_rollout: Iterable[LifecycleHandler] = (),
        cleanup_group: Iterable[LifecycleHandler] = (),
        teardown: Iterable[LifecycleHandler] = (),
    ):
        self.render_rollout_handlers = sorted_handlers("render", render_rollout)
        self.render_group_handlers = sorted_handlers("render", render_group)
        self.cleanup_rollout_handlers = sorted_handlers("cleanup", cleanup_rollout)
        self.cleanup_group_handlers = sorted_handlers("cleanup", cleanup_group)
        self.teardown_handlers = sorted_handlers("teardown", teardown)

    async def render_rollout(
        self, task: Task, state: State, resources: object
    ) -> State:
        await run_rollout_handlers(
            self.render_rollout_handlers,
            task=task,
            state=state,
            resources=resources,
        )
        return state

    async def cleanup_rollout(
        self, task: Task, state: State, resources: object
    ) -> State:
        await run_rollout_handlers(
            self.cleanup_rollout_handlers,
            task=task,
            state=state,
            resources=resources,
        )
        return state

    async def render_group(
        self, tasks: list[Task], states: list[State], resources: object
    ) -> list[State]:
        await run_group_handlers(
            self.render_group_handlers,
            tasks=tasks,
            states=states,
            resources=resources,
        )
        return states

    async def cleanup_group(
        self, tasks: list[Task], states: list[State], resources: object
    ) -> list[State]:
        await run_group_handlers(
            self.cleanup_group_handlers,
            tasks=tasks,
            states=states,
            resources=resources,
        )
        return states

    async def teardown(self) -> None:
        for handler in self.teardown_handlers:
            await maybe_call_with_named_args(handler)


async def run_rollout_handlers(
    handlers: Iterable[LifecycleHandler],
    *,
    task: Task,
    state: State,
    resources: object,
) -> None:
    for handler in handlers:
        await maybe_call_with_named_args(
            handler,
            task=task,
            state=state,
            resources=resources,
        )


async def run_group_handlers(
    handlers: Iterable[LifecycleHandler],
    *,
    tasks: list[Task],
    states: list[State],
    resources: object,
) -> None:
    for handler in handlers:
        await maybe_call_with_named_args(
            handler,
            tasks=tasks,
            states=states,
            resources=resources,
        )


def sorted_handlers(
    kind: str, handlers: Iterable[LifecycleHandler]
) -> list[LifecycleHandler]:
    priority_attr = f"{kind}_priority"
    return sorted(
        unique_handlers(list(handlers)),
        key=lambda handler: (
            -getattr(handler, priority_attr, 0),
            str(getattr(handler, "__name__", "")),
        ),
    )


def unique_handlers(handlers: list[LifecycleHandler]) -> list[LifecycleHandler]:
    unique: list[LifecycleHandler] = []
    seen: set[tuple[int, int]] = set()
    for handler in handlers:
        key = (
            id(getattr(handler, "__self__", None)),
            id(getattr(handler, "__func__", handler)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(handler)
    return unique
