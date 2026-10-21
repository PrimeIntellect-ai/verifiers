"""Durable agent calls, found by their traces.

With a `Durable` installed (`use_durable`, or `Env.durable` for an episode), every agent
call (`Agent.run`, `Agent.interaction`, `Agent.provision`) is durable. Its identity is
the durable's scope (`match`: for an episode, the env's `trace_labels`), the agent's name
(its seat) and its position: the key `<seat>/<n>` names the agent's n-th call since the
durable was installed (taken when the call starts, so `asyncio.gather` keeps the order).
A call may name its key instead (`key="draft"`: `<seat>/draft`, for a call order that is
not stable); `durable=False` opts a call out. The trace it makes is labelled with the key
and a fingerprint of what its result depends on (`trace.info["labels"]["call"]`), and
with its ref, a hash of its scope and key (`call_ref`). Before it runs, the latest earlier
trace carrying the same ref is looked up in the run's trace store (`TraceLookup.get`):

- a finished ok trace with the same fingerprint returns as it is, without running
  (`Trace.kept`);
- an unfinished or failed one with checkpoints and the same fingerprint is continued
  from its latest checkpoint (`resume=`, on `RETRY_NOTE`);
- anything else (nothing found, a changed fingerprint) runs fresh.

Run again after a crash, the same code makes the same calls in the same order, since the
results it saw are the recorded ones: each finds its own trace. Code changed so that
positions shift meets traces whose fingerprints differ, and those calls run fresh: safe,
never stale. Host code between agent calls must therefore be cheap and deterministic
given their results: anything expensive or nondeterministic is an agent run of its own
(a judge is an agent with the null harness), randomness comes from `episode_random()`,
and an external side effect is idempotent or left to the caller's commit. Without a
durable installed, calls are plain, exactly as before.

The trace is the record: there is no store of its own. What verifiers ships is the
lookup protocol (`TraceLookup`), the labels and refs, and the resolution; the store is
the run's (prime-rl writes every trace call by call, checkpoints included, and keeps a
pointer per ref to the trace carrying it that moved last). `TraceDir` is a directory of
traces for tests and standalone use.

    durable = vf.Durable(lookup, match={"flow": {"job": "j1"}})
    with vf.use_durable(durable):            # or Env.durable(task) for an episode
        trace = await agent.run(task)        # durable as "agent/0"
        other = await agent.run(task, key="solve")   # as "agent/solve"

`Durable.match` is the scope: the labels that, with the key, make the call's ref. Left
None, an episode's durable takes the env's `trace_labels(task)` (which every trace of
the episode carries); an env whose labels hold something that changes between attempts
of the same episode (an attempt number) names the scope without it. The durable stamps
`match` on the traces of its calls too. The fingerprint hashes the task's class, code,
data and config, the agent's model, sampling, harness and client, the durable's `parts`
(a pipeline version) and the call's `depends`. Budgets (timeouts, turn and token limits,
retries, saves) are not parts: raising a timeout keeps a finished call, while a changed
prompt, model or harness reruns it.

A continuation (`agent.run(resume=trace, note=...)`, `agent.interaction(resume=trace)`)
is a durable call too, by its position or `key`, so a stage that continues a seat's
conversation from an earlier stage keeps its own turns across a crash. Its fingerprint
holds the trace it continues (its origin) by id, with `at` and `note`
(`resume_part`): a finished continuation is kept, a cut-off one continues from its own
latest checkpoint (not the origin's), and a continuation of another origin (the earlier
stage ran again) or with another note runs fresh from that origin.

A box from `agent.provision(task)` is durable the same way: every run on it is
labelled with the box's key, its ref (`box_ref`: the scope, key and fingerprint) and the
checkpoint the box was at when the run started (`labels["box"]`), and a later provision
under the key starts its box from the latest checkpoint of the run on it that moved last
(`box_checkpoint`, see `Box`).
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import json
import logging
import os
import random
import tempfile
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from verifiers.v1.task import identity_data, source_hash
from verifiers.v1.trace import (
    LABELS_KEY,
    Checkpoint,
    Trace,
    WireTrace,
)

if TYPE_CHECKING:
    from verifiers.v1.agent import Agent
    from verifiers.v1.task import Task

logger = logging.getLogger(__name__)

CALL_LABEL = "call"
"""The label a durable call stamps: `{"key", "fingerprint", "parts", "ref"}`."""
BOX_LABEL = "box"
"""The label every run on a durable box carries: `{"key", "fingerprint", "ref"}`, and
`"checkpoint": {"id", "time"}` once the box has one (`Box.labels`)."""

FRAMEWORK_PACKAGES = ("verifiers", "pydantic", "builtins", "typing", "abc")
"""Packages whose classes are framework code: a task class's code hash skips them, so
upgrading verifiers keeps every finished call."""


@runtime_checkable
class TraceLookup(Protocol):
    """Where a durable call finds its earlier traces: the run's trace store."""

    def get(self, ref: str) -> Trace | None:
        """Of the stored traces carrying `ref` (`refs_of` their labels), finished or not,
        the one that moved last (`recency`: its latest checkpoint's time, else its
        start); a partial one as of its latest save. None when there is none. A store
        keeps a pointer per ref, moved as the traces carrying it are saved, so this is
        one read, not a search."""
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
    match: Mapping[str, Any] | None = None
    """Labels an earlier trace must carry, beyond the call's key: the scope (the
    episode's identity). The traces of this durable's calls are stamped with them.
    None: an episode's `Env.trace_labels(task)`; outside an episode, no labels."""
    parts: Mapping[str, str] = field(default_factory=dict)
    """Fingerprint parts every call gets (e.g. `{"version": pipeline.version}`)."""
    framework: tuple[str, ...] = ()
    """Packages whose task classes count as framework code beyond `FRAMEWORK_PACKAGES`:
    editing them does not rerun a call."""

    @property
    def scope(self) -> dict[str, Any]:
        """The scope's labels (`match`), a copy."""
        return copy.deepcopy(dict(self.match or {}))

    def within(self, match: Mapping[str, Any]) -> Durable:
        """This durable with another scope: `match` replaces the scope's labels."""
        return replace(self, match=copy.deepcopy(dict(match)))

    def get(self, ref: str) -> Trace | None:
        """The trace `ref` points at (`TraceLookup.get`), if it carries the ref."""
        found = self.lookup.get(ref)
        return found if found is not None and ref in refs_of(found.labels) else None

    def write(self, trace: Trace) -> None:
        if isinstance(self.lookup, TraceSink):
            try:
                self.lookup.write(trace)
            except StaleTraceError:
                raise
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
    position and is keyed `<name>/<key>`. A key with no durable at all is a plain call,
    labelled with its key."""
    if durable is False:
        return None, None
    found = durable if isinstance(durable, Durable) else current_durable()
    if key is not None:
        if not key or key.isdigit():
            raise ValueError(
                f"key={key!r} would stand for a position; name the call (e.g. 'solve')"
            )
        return found, f"{name}/{key}"
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


_MIXIN_HASHES: weakref.WeakKeyDictionary[type, str | None] = weakref.WeakKeyDictionary()


def _class_hash(cls: type) -> str | None:
    """The source hash of `cls` as it was loaded. `inspect.getsource` reads the file
    on disk, so a hash taken later follows the checkout, not the running code. A
    `vf.Task` subclass stores its hash at class creation (`__source_hash__`); any
    other class is hashed once per process, on first use."""
    if "__source_hash__" in cls.__dict__:
        return cls.__dict__["__source_hash__"]
    if cls not in _MIXIN_HASHES:
        _MIXIN_HASHES[cls] = source_hash(cls)
    return _MIXIN_HASHES[cls]


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
        parts.append(f"{base.__module__}:{base.__qualname__}:{_class_hash(base)}")
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
        "data": _hash(identity_data(_dump(task.data))),
        "task_config": _hash(_dump(task.config)),
    }


def call_parts(
    task: Task,
    agent: Agent,
    durable: Durable | None,
    depends: Any = None,
    resume: Trace | None = None,
    at: str | int | None = None,
    note: str | None = None,
) -> dict[str, str]:
    """What a call's result depends on, hashed part by part: the task (`task_parts`),
    the agent (`agent_parts`), the durable's `parts` and the call's `depends`. A
    continuation (`resume`) adds the trace it continues, by id, with its `at` and
    `note` (`resume_part`)."""
    framework = durable.framework if durable is not None else ()
    parts = {**task_parts(task, framework), **agent_parts(agent)}
    if durable is not None:
        parts.update(durable.parts)
    if depends is not None:
        parts["depends"] = depends_hash(depends)
    if resume is not None:
        parts["resume"] = resume_part(resume, at, note)
    return parts


def resume_part(resume: Trace, at: str | int | None, note: str | None) -> str:
    """The fingerprint part of a continuation: the id of the trace it continues (its
    origin), the checkpoint it continues from (`at`, as given) and its `note`. Another
    origin (an earlier stage run again, or a later run of the trace: `Trace.attempts`)
    or another note runs the continuation fresh. A trace's first run adds no `run`, so
    the part is as it was before traces were resumed in place."""
    origin: dict[str, Any] = {"trace": resume.id, "at": at, "note": note}
    if resume.attempts > 1:
        origin["run"] = resume.attempts - 1
    return _hash(origin)


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


def call_ref(scope: Mapping[str, Any], key: str) -> str:
    """The address of durable call `key` in `scope` (`Durable.scope`): every trace of
    the call carries it (`labels["call"]["ref"]`), and the run's store points it at the
    one that moved last (`TraceLookup.get`)."""
    return _hash({"scope": scope, "call": key})


def box_ref(scope: Mapping[str, Any], key: str, box_fingerprint: str) -> str:
    """The address of durable box `key` in `scope` with `box_fingerprint`: every run on
    the box carries it (`labels["box"]["ref"]`)."""
    return _hash({"scope": scope, "box": key, "fingerprint": box_fingerprint})


class StaleTraceError(RuntimeError):
    """A save of a trace from a run that a later resume of it replaced (a stale
    attempt, a zombie process): the stored record holds more runs (`Trace.attempts`)."""


def refs_of(labels: Mapping[str, Any]) -> list[str]:
    """The refs a trace's labels carry: its durable call's and its durable box's."""
    found = []
    for name in (CALL_LABEL, BOX_LABEL):
        label = labels.get(name)
        if isinstance(label, Mapping) and isinstance(label.get("ref"), str):
            found.append(label["ref"])
    return found


def changed_parts(old: Mapping[str, Any], new: Mapping[str, str]) -> list[str]:
    return sorted(name for name in {*old, *new} if old.get(name) != new.get(name))


# -- resolution -----------------------------------------------------------------------


def recency(trace: Trace) -> float:
    """When a trace last moved: its start or its latest checkpoint, whichever is later
    (the order `TraceLookup.get` picks by)."""
    times = [trace.timing.boot.start or 0.0]
    times.extend(c.time for c in trace.checkpoints if c.time is not None)
    return max(times)


def latest(traces: Iterable[Trace]) -> Trace | None:
    """The one of `traces` that moved last (`recency`)."""
    return max(traces, key=recency, default=None)


@dataclass
class Resolved:
    """What a durable call does: return `kept`, continue `resume`, or run fresh."""

    label: dict[str, Any]
    kept: Trace | None = None
    resume: Trace | None = None


def resolve_call(
    durable: Durable | None,
    key: str,
    parts: dict[str, str],
) -> Resolved:
    """Look the call up: the latest earlier trace of `key` in the durable's scope (its
    `call_ref`) decides (see the module docs). Without a durable the call runs fresh."""
    label = {
        "key": key,
        "fingerprint": fingerprint(parts),
        "parts": parts,
        "ref": call_ref(durable.scope if durable is not None else {}, key),
    }
    if durable is None:
        return Resolved(label)
    previous = durable.get(label["ref"])
    if previous is None:
        return Resolved(label)
    old = previous.labels.get(CALL_LABEL) or {}
    if old.get("fingerprint") != label["fingerprint"]:
        changed = changed_parts(old.get("parts") or {}, parts) or ["fingerprint"]
        logger.info("call %r runs fresh: %s changed", key, ", ".join(changed))
        return Resolved(label)
    if previous.is_completed and previous.ok:
        kept = previous.model_copy(deep=True)
        kept._kept = True
        return Resolved(label, kept=kept)
    if previous.checkpoints:
        return Resolved(label, resume=previous)
    return Resolved(label)


def box_checkpoint(durable: Durable | None, ref: str) -> Checkpoint | None:
    """The latest checkpoint of the box `ref` (`box_ref`) names: of the latest run on it,
    its own latest checkpoint or the one the box was at when it started (`Box.labels`),
    whichever is later. A run that moved last holds the box's latest checkpoint either
    way: it took it, or the box was at it."""
    if durable is None:
        return None
    trace = durable.get(ref)
    if trace is None:
        return None
    found = [c for c in trace.checkpoints if c.time is not None]
    at = (trace.labels.get(BOX_LABEL) or {}).get("checkpoint")
    if isinstance(at, Mapping) and at.get("id"):
        found.append(
            Checkpoint(id=at["id"], turns=0, restored=True, time=at.get("time") or 0.0)
        )
    return max(found, key=lambda c: c.time or 0.0, default=None)


def episode_random(*salt: Any) -> random.Random:
    """A `random.Random` seeded from the episode's identity (the installed durable's
    scope) and `salt`: an episode run again after a crash draws the same numbers, so
    host code between its agent calls stays deterministic. Take one per episode (or
    per purpose, by `salt`) and draw from it in a fixed order. Without a durable
    installed, an unseeded one: nothing reruns."""
    durable = current_durable()
    if durable is None:
        return random.Random()
    seed = hashlib.sha256(
        json.dumps([durable.scope, list(salt)], sort_keys=True, default=str).encode()
    ).digest()
    return random.Random(int.from_bytes(seed[:8], "big"))


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
    """A directory of traces, one `<id>.json` record each, with a pointer per ref
    (`refs/<ref>.json`): a `TraceLookup` and a `TraceSink` for tests and standalone use.
    The pointer is the call's record: the trace and which of its runs is the call's
    (`Trace.attempts`), so a call whose trace a later call resumed in place gets its
    own run back (`Trace.attempt`). Each write replaces the trace's file atomically
    and moves the pointers of its refs to it unless they point at a trace that moved
    later. A write holding fewer runs than the stored trace is from a run a later
    resume replaced, and is refused (`StaleTraceError`)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def write(self, trace: Trace) -> None:
        runs = trace.attempts
        held = _read(self.path / "runs" / f"{trace.id}.json")
        if held is not None and held["runs"] > runs:
            raise StaleTraceError(
                f"trace {trace.id} holds {held['runs']} runs; this save is from a "
                f"replaced run ({runs})"
            )
        _replace(self.path / f"{trace.id}.json", trace.to_record())
        if runs > 1:
            _replace(self.path / "runs" / f"{trace.id}.json", {"runs": runs})
        moved = recency(trace)
        for ref in refs_of(trace.labels):
            pointer = self.path / "refs" / f"{ref}.json"
            held = _read(pointer)
            if held is None or held["trace"] == trace.id or held["recency"] <= moved:
                _replace(
                    pointer, {"trace": trace.id, "run": runs - 1, "recency": moved}
                )

    def get(self, ref: str) -> Trace | None:
        held = _read(self.path / "refs" / f"{ref}.json")
        if held is None:
            return None
        try:
            trace = WireTrace.model_validate_json(
                (self.path / f"{held['trace']}.json").read_bytes()
            )
        except (OSError, ValueError):
            return None
        run = held.get("run")
        return trace if run is None else trace.attempt(min(run, trace.attempts - 1))


def _replace(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(record, default=str))
    os.replace(tmp, path)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
