import inspect
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, cast

from verifiers.utils.async_utils import maybe_call_with_named_args

from .config_callable_utils import CallableKind
from ..state import State
from ..types import Handler

if TYPE_CHECKING:
    from ..harness import Harness
    from ..taskset import Taskset
    from ..toolset import Toolset
    from ..user import User

LifecycleStage = Literal["rollout", "group"]


def collect_handlers(
    owners: Iterable["Taskset | Harness | Toolset | User | None"],
    attr: str,
    extra: Iterable[Handler] = (),
    stage: LifecycleStage | None = None,
) -> list[Handler]:
    handlers: list[Handler] = []
    for owner in owners:
        if owner is None:
            continue
        for _, method in inspect.getmembers(owner, predicate=callable):
            if handler_is_marked(method, cast(CallableKind, attr)):
                handlers.append(cast(Handler, method))
    handlers.extend(extra)
    if stage is not None:
        handlers = [
            handler
            for handler in handlers
            if handler_stage(handler, cast(CallableKind, attr)) == stage
        ]
    return sort_handlers(unique_handlers(handlers), attr)


def validate_handler_args(
    handlers: Iterable[Handler],
    expected: set[str],
    attr: str,
    stage: LifecycleStage,
) -> None:
    context = f"{stage} {attr}"
    for handler in handlers:
        signature = inspect.signature(handler)
        for parameter in signature.parameters.values():
            if parameter.kind == parameter.POSITIONAL_ONLY:
                raise TypeError(
                    f"{context} handler {handler!r} must use named parameters."
                )
            if (
                parameter.kind == parameter.VAR_POSITIONAL
                and parameter.name not in expected
            ):
                raise TypeError(f"{context} handler {handler!r} must not use *args.")


async def run_handlers(handlers: Iterable[Handler], **kwargs: object) -> None:
    for handler in handlers:
        await maybe_call_with_named_args(handler, **kwargs)


def unique_handlers(
    handlers: Iterable[Handler],
) -> list[Handler]:
    unique: list[Handler] = []
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


def sort_handlers(handlers: Iterable[Handler], attr: str) -> list[Handler]:
    return sorted(
        handlers,
        key=lambda handler: (
            -int(getattr(handler, f"{attr}_priority", 0)),
            str(getattr(handler, "__name__", "")),
        ),
    )


async def state_done(state: State) -> bool:
    return bool(state.get("done"))


def handler_is_marked(handler: object, kind: CallableKind) -> bool:
    return getattr(handler, kind, False) is True


def handler_stage(handler: object, kind: CallableKind) -> LifecycleStage:
    return cast(LifecycleStage, getattr(handler, f"{kind}_stage", "rollout"))
