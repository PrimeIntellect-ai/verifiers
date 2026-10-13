"""Durable agent calls, found by their traces.

With a `Durable` installed (`use_durable`, or `Env.durable` for an episode), every agent
call (`Agent.run`, `Agent.interaction`, `Agent.provision`) is durable. Its key is its
position, `<agent name>/<n>` for the agent's n-th call since the durable was installed
(taken when the call starts, so `asyncio.gather` keeps the order), unless it names one
(`key=`, for a call order that is not stable); `durable=False` opts a call out. The trace
it makes is labelled with the key and a fingerprint of what its result depends on
(`trace.info["labels"]["call"]`), and before it runs, the newest earlier trace carrying
the same key is looked up in the run's trace store:

- a finished ok trace with the same fingerprint returns as it is, without running
  (`Trace.kept`);
- an unfinished or failed one with checkpoints and the same fingerprint is continued
  from its latest checkpoint (`resume=`, on `RETRY_NOTE`);
- anything else (nothing found, a changed fingerprint) runs fresh.

Run again after a crash, the same code makes the same calls in the same order, since the
results it saw are the recorded ones: each finds its own trace. Code changed so that
positions shift meets traces whose fingerprints differ, and those calls run fresh: safe,
never stale. Host code between agent calls that is nondeterministic or has side effects
must therefore be idempotent, or go through a durable step of its own. Without a durable
installed, calls are plain, exactly as before.

The trace is the record: there is no store of its own. What verifiers ships is the
lookup protocol (`TraceLookup`), the labels, and the resolution; the store is the
run's (prime-rl writes every trace call by call, checkpoints included). `TraceDir` is
a directory of traces for tests and standalone use.

    durable = vf.Durable(lookup, match={"flow": {"job": "j1"}})
    with vf.use_durable(durable):            # or Env.durable(task) for an episode
        trace = await agent.run(task)        # durable as "agent/0"
        other = await agent.run(task, key="solve")

`Durable.match` is the scope: the labels an earlier trace must carry to count (an env
stamps its `trace_labels` on every trace; the durable stamps `match` on the traces of
its calls too, so a standalone caller finds them). The fingerprint hashes the task's
class, code, data and config, the agent's model, sampling, harness and client, the
durable's `parts` (a pipeline version) and the call's `depends`; `cache_inputs`
replaces the task and agent parts. Budgets (timeouts, turn and token limits, retries,
saves) are not parts: raising a timeout keeps a finished call, while a changed prompt,
model or harness reruns it.

A box from `agent.provision(task)` is durable the same way: every run on it is
labelled with the box's key (`labels["box"]`), and a later provision under the key
starts its box from the latest checkpoint any of those runs recorded (see `Box`).
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import inspect
import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from verifiers.v1.trace import LABELS_KEY, Checkpoint, Trace, WireTrace

if TYPE_CHECKING:
    from verifiers.v1.agent import Agent
    from verifiers.v1.task import Task

logger = logging.getLogger(__name__)

CALL_LABEL = "call"
"""The label a durable call stamps: `{"key", "fingerprint", "parts"}`."""
BOX_LABEL = "box"
"""The label every run on a durable box carries: `{"key", "fingerprint"}`."""

FRAMEWORK_PACKAGES = ("verifiers", "pydantic", "builtins", "typing", "abc")
"""Packages whose classes are framework code: a task class's code hash skips them, so
upgrading verifiers keeps every finished call."""


@runtime_checkable
class TraceLookup(Protocol):
    """Where a durable call finds its earlier traces: the run's trace store."""

    def find(self, labels: Mapping[str, Any]) -> Iterable[Trace]:
        """Every stored trace whose labels (`trace.info["labels"]`) contain `labels`
        (`labels_match`), finished or not; a partial one as of its latest save. Order
        does not matter. Called once per durable call, so it should not rescan the
        whole store each time."""
        ...


@runtime_checkable
class TraceSink(Protocol):
    """A lookup that is also the store: durable calls write their traces to it on
    every per-call save and when they finish (`TraceDir`). A lookup over a store
    someone else writes (prime-rl's) does not implement this."""

    def write(self, trace: Trace) -> None: ...


@dataclass(frozen=True)
class Durable:
    """How durable calls resolve: where their traces are (`lookup`), which of them
    count (`match`, the scope), and what else every fingerprint holds (`parts`)."""

    lookup: TraceLookup
    match: Mapping[str, Any] = field(default_factory=dict)
    """Labels an earlier trace must carry, beyond the call's key: the scope. The
    traces of this durable's calls are stamped with them."""
    parts: Mapping[str, str] = field(default_factory=dict)
    """Fingerprint parts every call gets (e.g. `{"version": pipeline.version}`)."""
    framework: tuple[str, ...] = ()
    """Packages whose task classes count as framework code beyond `FRAMEWORK_PACKAGES`:
    editing them does not rerun a call."""

    def within(self, match: Mapping[str, Any]) -> Durable:
        """This durable with another scope: `match` replaces the scope's labels."""
        return replace(self, match=copy.deepcopy(dict(match)))

    def find(self, labels: Mapping[str, Any]) -> list[Trace]:
        query = merge_labels(copy.deepcopy(dict(self.match)), labels)
        return [t for t in self.lookup.find(query) if labels_match(t.labels, query)]

    def write(self, trace: Trace) -> None:
        if isinstance(self.lookup, TraceSink):
            try:
                self.lookup.write(trace)
            except Exception:
                logger.warning(
                    "durable: writing trace %s failed", trace.id, exc_info=True
                )


@dataclass
class _Installed:
    durable: Durable
    positions: dict[str, int] = field(default_factory=dict)
    """Per agent name, how many calls took a position so far."""


_CURRENT: contextvars.ContextVar[_Installed | None] = contextvars.ContextVar(
    "vf_durable", default=None
)


def current_durable() -> Durable | None:
    """The durable installed for the running task (`use_durable`, `Env.durable`)."""
    installed = _CURRENT.get()
    return installed.durable if installed is not None else None


@contextlib.contextmanager
def use_durable(value: Durable | None) -> Iterator[Durable | None]:
    """Install `value` for the calls made in this context (and in tasks it starts):
    every agent call in it is durable, under its position unless it names a key. The
    positions start over here. None turns durability off inside it."""
    token = _CURRENT.set(_Installed(value) if value is not None else None)
    try:
        yield value
    finally:
        _CURRENT.reset(token)


def plan_call(
    name: str, key: str | None, durable: Durable | bool | None
) -> tuple[Durable | None, str | None]:
    """Which durable a call resolves through and under which key; `(None, None)` for
    a plain call. `durable=False` opts the call out. Without a key the call takes the
    next position of its agent (`<name>/<n>`, counted from 0 since the durable was
    installed); it is taken when the call starts, before it awaits anything, so calls
    started together (`asyncio.gather`) keep their order. A call with a key takes no
    position. A key with no durable at all is a plain call, labelled with its key."""
    if durable is False:
        return None, None
    found = durable if isinstance(durable, Durable) else current_durable()
    if key is not None:
        return found, key
    installed = _CURRENT.get()
    if installed is None:
        if found is not None:
            raise ValueError(
                "a call given durable= without key= takes a position, which needs a "
                "durable installed (use_durable, Env.durable)"
            )
        return None, None
    position = installed.positions.get(name, 0)
    installed.positions[name] = position + 1
    return found, f"{name}/{position}"


# -- labels ---------------------------------------------------------------------------


def labels_match(labels: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    """Whether `labels` contain `query`: every key of `query` is in `labels` with an
    equal value, nested mappings compared the same way (a subset)."""
    for name, wanted in query.items():
        if name not in labels:
            return False
        have = labels[name]
        if isinstance(wanted, Mapping):
            if not isinstance(have, Mapping) or not labels_match(have, wanted):
                return False
        elif have != wanted:
            return False
    return True


def merge_labels(into: dict[str, Any], labels: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `labels` into `into` in place (nested mappings merged, other values
    replaced) and return it."""
    for name, value in labels.items():
        if isinstance(value, Mapping) and isinstance(into.get(name), dict):
            merge_labels(into[name], value)
        else:
            into[name] = copy.deepcopy(value)
    return into


def stamp(trace: Trace, labels: Mapping[str, Any], drop: Iterable[str] = ()) -> None:
    """Merge `labels` into the trace's labels, dropping the names in `drop` first (a
    continuation inherits its source's labels, some of which no longer hold)."""
    if LABELS_KEY not in trace.info and not labels:
        return
    current = trace.info.get(LABELS_KEY)
    current = dict(current) if isinstance(current, Mapping) else {}
    for name in drop:
        current.pop(name, None)
    trace.info[LABELS_KEY] = merge_labels(current, labels)


# -- fingerprints ---------------------------------------------------------------------


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


def _hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def _source_hash(obj: Any) -> str | None:
    try:
        return hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return None


def code_hash(cls: type, framework: Iterable[str] = ()) -> str:
    """The code version of a task class: its `__code_version__` when it sets one (edits
    rerun nothing until it changes), else a hash of the source of it and its base
    classes outside the framework (`FRAMEWORK_PACKAGES` and `framework`). A class
    without source hashes as its name."""
    pinned = getattr(cls, "__code_version__", None)
    if pinned is not None:
        return f"pinned:{pinned}"
    skip = (*FRAMEWORK_PACKAGES, *framework)
    parts = []
    for base in cls.__mro__:
        package = base.__module__.split(".")[0]
        if package in skip:
            continue
        parts.append(f"{base.__module__}:{base.__qualname__}:{_source_hash(base)}")
    return _hash(parts)


def depends_hash(depends: Any) -> str:
    """Hash of a call's extra inputs (`depends=`): JSON values as they are, a `Path` by
    its file contents (a directory by every file under it, by relative path)."""

    def resolve(value: Any) -> Any:
        if isinstance(value, Path):
            return _path_hash(value)
        if isinstance(value, Mapping):
            return {str(k): resolve(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [resolve(v) for v in value]
        return _dump(value)

    return _hash(resolve(depends))


def _path_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            relative = str(file.relative_to(path)).encode()
            digest.update(relative + b"\0" + file.read_bytes() + b"\0")
    elif path.exists():
        digest.update(path.read_bytes())
    else:
        digest.update(b"missing:" + str(path).encode())
    return "file:" + digest.hexdigest()[:16]


_CLIENT_FIELDS = ("type", "base_url", "renderer", "renderer_model_name")
_HARNESS_BUDGETS = {"tool_timeout"}


def agent_parts(agent: Agent) -> dict[str, str]:
    """What an agent contributes to a call's fingerprint: its model, sampling, harness
    (less its tool timeout) and client (type, base_url, renderer). Budgets are left
    out: timeouts, turn and token limits, retries, saves, credentials."""
    config = agent.config
    harness = config.harness
    client = config.client
    return {
        "model": _hash(config.model),
        "sampling": _hash(_dump(config.sampling)),
        "harness": _hash(
            None
            if harness is None
            else harness.model_dump(mode="json", exclude=_HARNESS_BUDGETS)
        ),
        "client": _hash(
            None
            if client is None
            else {name: _dump(getattr(client, name, None)) for name in _CLIENT_FIELDS}
        ),
    }


def task_parts(task: Task, framework: Iterable[str] = ()) -> dict[str, str]:
    cls = type(task)
    return {
        "task": f"{cls.__module__}:{cls.__qualname__}",
        "task_code": code_hash(cls, framework),
        "data": _hash(_dump(task.data)),
        "task_config": _hash(_dump(task.config)),
    }


def call_parts(
    task: Task,
    agent: Agent,
    durable: Durable | None,
    depends: Any = None,
    cache_inputs: Any = None,
) -> dict[str, str]:
    """What a call's result depends on, hashed part by part: the task (`task_parts`),
    the agent (`agent_parts`), the durable's `parts` and the call's `depends`.
    `cache_inputs`, when given, replaces the task and agent parts with its own hash
    (`{}`: nothing but the durable's parts and `depends`)."""
    framework = durable.framework if durable is not None else ()
    if cache_inputs is not None:
        parts = {"inputs": _hash(cache_inputs)}
    else:
        parts = {**task_parts(task, framework), **agent_parts(agent)}
    if durable is not None:
        parts.update(durable.parts)
    if depends is not None:
        parts["depends"] = depends_hash(depends)
    return parts


def box_parts(
    task: Task | None, agent: Agent, durable: Durable | None, depends: Any = None
) -> dict[str, str]:
    """What a durable box's state depends on: the task it was provisioned for and the
    agent's runtime policy, the durable's `parts` and `depends`."""
    framework = durable.framework if durable is not None else ()
    parts = task_parts(task, framework) if task is not None else {}
    parts["runtime"] = _hash(_dump(agent.runtime_config))
    if durable is not None:
        parts.update(durable.parts)
    if depends is not None:
        parts["depends"] = depends_hash(depends)
    return parts


def fingerprint(parts: Mapping[str, str]) -> str:
    return _hash(dict(parts))


def changed_parts(old: Mapping[str, Any], new: Mapping[str, str]) -> list[str]:
    return sorted(name for name in {*old, *new} if old.get(name) != new.get(name))


# -- resolution -----------------------------------------------------------------------


def recency(trace: Trace) -> float:
    """When a trace last moved: its start or its latest checkpoint, whichever is later."""
    times = [trace.timing.boot.start or 0.0]
    times.extend(c.time for c in trace.checkpoints if c.time is not None)
    return max(times)


def latest(traces: Iterable[Trace]) -> Trace | None:
    """The newest of `traces` (`recency`)."""
    return max(traces, key=recency, default=None)


@dataclass
class Resolved:
    """What a durable call does: return `kept`, continue `resume`, or run fresh."""

    label: dict[str, Any]
    kept: Trace | None = None
    resume: Trace | None = None
    previous: Trace | None = None
    """The newest earlier trace of the key, whatever became of it."""


def resolve_call(
    durable: Durable | None,
    key: str,
    parts: dict[str, str],
) -> Resolved:
    """Look the call up: the newest earlier trace of `key` in the durable's scope
    decides (see the module docs). Without a durable the call runs fresh."""
    label = {"key": key, "fingerprint": fingerprint(parts), "parts": parts}
    if durable is None:
        return Resolved(label)
    previous = latest(durable.find({CALL_LABEL: {"key": key}}))
    if previous is None:
        return Resolved(label)
    old = previous.labels.get(CALL_LABEL) or {}
    if old.get("fingerprint") != label["fingerprint"]:
        changed = changed_parts(old.get("parts") or {}, parts) or ["fingerprint"]
        logger.info("call %r runs fresh: %s changed", key, ", ".join(changed))
        return Resolved(label, previous=previous)
    if previous.is_completed and previous.ok:
        kept = previous.model_copy(deep=True)
        kept._kept = True
        return Resolved(label, kept=kept, previous=previous)
    if previous.checkpoints:
        return Resolved(label, resume=previous, previous=previous)
    return Resolved(label, previous=previous)


def box_checkpoint(
    durable: Durable | None, key: str, box_fingerprint: str
) -> Checkpoint | None:
    """The latest checkpoint any run on box `key` recorded (by `Checkpoint.time`), from
    the runs whose box label carries the same fingerprint."""
    if durable is None:
        return None
    found: list[Checkpoint] = []
    for trace in durable.find({BOX_LABEL: {"key": key}}):
        label = trace.labels.get(BOX_LABEL) or {}
        if label.get("fingerprint") == box_fingerprint:
            found.extend(c for c in trace.checkpoints if c.time is not None)
    return max(found, key=lambda c: c.time or 0.0, default=None)


def observe(
    on_trace: Callable[[Trace], None] | None,
    labels: Mapping[str, Any],
    drop: Iterable[str] = (),
) -> Callable[[Trace], None]:
    """An `on_trace` that stamps `labels` on each minted trace before `on_trace` sees it."""
    drop = tuple(drop)

    def minted(trace: Trace) -> None:
        stamp(trace, labels, drop)
        if on_trace is not None:
            on_trace(trace)

    return minted


# -- a directory of traces ------------------------------------------------------------


class TraceDir:
    """A directory of traces, one `<id>.json` record each: a `TraceLookup` and a
    `TraceSink` for tests and standalone use. Each write replaces the trace's file
    atomically. `find` reads the files it has not read since they last changed."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._cache: dict[Path, tuple[float, int, Trace | None]] = {}

    def write(self, trace: Trace) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        data = json.dumps(trace.to_record(), default=str)
        fd, tmp = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, self.path / f"{trace.id}.json")

    def _load(self, file: Path) -> Trace | None:
        try:
            stat = file.stat()
        except FileNotFoundError:
            return None
        cached = self._cache.get(file)
        if cached is not None and cached[:2] == (stat.st_mtime, stat.st_size):
            return cached[2]
        try:
            trace = WireTrace.model_validate_json(file.read_bytes())
        except (OSError, ValueError):
            trace = None
        self._cache[file] = (stat.st_mtime, stat.st_size, trace)
        return trace

    def find(self, labels: Mapping[str, Any]) -> list[Trace]:
        if not self.path.is_dir():
            return []
        found = []
        for file in sorted(self.path.glob("*.json")):
            trace = self._load(file)
            if trace is not None and labels_match(trace.labels, labels):
                found.append(trace)
        return found
