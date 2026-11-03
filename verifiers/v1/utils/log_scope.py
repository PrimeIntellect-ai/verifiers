"""Scope prefixes for the library's log lines.

An episode tags every line logged inside it `[<episode>]`; an agent run inside an
episode extends that to `[<episode> <role>#<n>]`. The prefix comes from contextvars,
so call sites log plain messages and asyncio tasks inherit the scope they start in.
Ids are shortened to `ID_CHARS`: a prefix of the full id, so it still greps
against stored traces.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

ID_CHARS = 12

_episode: ContextVar[str | None] = ContextVar("vf_log_episode", default=None)
_rollout: ContextVar[str | None] = ContextVar("vf_log_rollout", default=None)

LogScope = tuple[str | None, str | None]


class _ScopeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        parts = [part for part in (_episode.get(), _rollout.get()) if part]
        if parts:
            record.msg = f"[{' '.join(parts)}] {record.getMessage()}"
            record.args = ()
        return True


_FILTER = _ScopeFilter()


def get_logger(name: str) -> logging.Logger:
    """The module logger for `name`, prefixing records with the current scope."""
    logger = logging.getLogger(name)
    if _FILTER not in logger.filters:
        logger.addFilter(_FILTER)
    return logger


def short_id(id: str) -> str:
    return id[:ID_CHARS]


def seconds(value: float) -> str:
    return f"{value:.1f}s"


def clip(message: str, limit: int = 200) -> str:
    """The first line of an error message, short enough for a log line."""
    line = message.strip().splitlines()[0] if message.strip() else ""
    return line if len(line) <= limit else f"{line[: limit - 3]}..."


def current_scope() -> LogScope:
    return _episode.get(), _rollout.get()


@contextmanager
def scope(*, episode: str | None = None, rollout: str | None = None) -> Iterator[None]:
    """Tag lines logged inside with an episode and/or agent-run scope."""
    tokens = []
    if episode is not None:
        tokens.append((_episode, _episode.set(episode)))
    if rollout is not None:
        tokens.append((_rollout, _rollout.set(rollout)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def enter_scope(saved: LogScope) -> None:
    """Enter a scope captured with `current_scope()` for the rest of the current task
    (e.g. a server handler serving that rollout). Wrap the task's work in `isolated()`
    when the task serves more than one rollout."""
    episode, rollout = saved
    _episode.set(episode)
    _rollout.set(rollout)


@contextmanager
def isolated() -> Iterator[None]:
    """Undo any scope entered inside on exit."""
    tokens = [(_episode, _episode.set(None)), (_rollout, _rollout.set(None))]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
