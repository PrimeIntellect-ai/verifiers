"""The episode — one run's traces plus their shared standing, whole."""

import uuid
from typing import Generic

from pydantic import Field

from verifiers.v1.state import State, StateT
from verifiers.v1.task import DataT, WireTaskData
from verifiers.v1.trace import Error, Trace
from verifiers.v1.types import StrictBaseModel


class Episode(StrictBaseModel, Generic[DataT, StateT]):
    """One run of a task, whole: its identity and standing (`id`, `env`, `errors`)
    next to its flat `traces` — the object `finalize()` receives, the engine
    returns, and the durability envelope: one episode is one `traces.jsonl` line
    and one serve reply, so it persists and arrives whole or not at all — a torn
    line is the whole episode owed again, and a failure before any trace minted
    still leaves its errors here. Episode standing lives ONLY here (zero
    redundancy on the traces); per-trace facts (`agent`, per-trace errors) stay
    on the traces, which remain the atomic unit.

    `errors` are failures not attributable to any one trace (the env's
    `run`/`finalize` hooks, plus prior attempts' when retried).

    The type parameters serve the wire loaders: `WireEpisode` reads any taskset's
    episodes without importing the taskset."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    env: str = ""
    """The env that ran the episode (`EnvConfig.env_id`, e.g.
    `agentic-judge+gsm8k-v1`)."""
    ok: bool = False
    """THE success sentinel — the resume unit's keep-verdict, stamped by the
    engine when the final attempt's hooks and every trace concluded clean.
    Distinct from `errors` emptiness: a retried-and-recovered episode is `ok`
    and still keeps its earlier attempts' errors."""
    errors: list[Error] = Field(default_factory=list)
    traces: list[Trace[DataT, StateT]] = Field(default_factory=list)

    @property
    def error(self) -> Error | None:
        return self.errors[-1] if self.errors else None

    @classmethod
    def of(cls, trace: Trace, env: str = "") -> "Episode":
        """The single-agent record: one trace as its own episode."""
        return cls(env=env, traces=[trace], ok=trace.ok)


WireEpisode = Episode[WireTaskData, State]
"""Record loader that preserves unknown task fields in `task.model_extra`."""
