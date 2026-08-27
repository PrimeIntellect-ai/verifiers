"""Optional ACP lineage carried beside the training message graph.

The message DAG remains the source of training branches. This module describes the
runtime provenance that explains which recursive session and context epoch produced each
model call. ACP agents send the call-local part in private HTTP headers and publish the
full manifest through response ``_meta``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LINEAGE_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"

ACP_LINEAGE_METADATA_KEY = "ai.prime.acp/lineage-v1"
ACP_REQUEST_ID_HEADER = "X-ACP-Lineage-Request-ID"
ACP_SESSION_ID_HEADER = "X-ACP-Lineage-Session-ID"
ACP_PARENT_SESSION_ID_HEADER = "X-ACP-Lineage-Parent-Session-ID"
ACP_CONTEXT_ID_HEADER = "X-ACP-Lineage-Context-ID"
ACP_PREVIOUS_CONTEXT_ID_HEADER = "X-ACP-Lineage-Previous-Context-ID"
ACP_TRANSITION_HEADER = "X-ACP-Lineage-Transition"
ACP_COMPACTION_ID_HEADER = "X-ACP-Lineage-Compaction-ID"
ACP_DEPTH_HEADER = "X-ACP-Lineage-Depth"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"

ACP_LINEAGE_HEADERS = frozenset(
    {
        ACP_REQUEST_ID_HEADER.lower(),
        ACP_SESSION_ID_HEADER.lower(),
        ACP_PARENT_SESSION_ID_HEADER.lower(),
        ACP_CONTEXT_ID_HEADER.lower(),
        ACP_PREVIOUS_CONTEXT_ID_HEADER.lower(),
        ACP_TRANSITION_HEADER.lower(),
        ACP_COMPACTION_ID_HEADER.lower(),
        ACP_DEPTH_HEADER.lower(),
    }
)
"""Private ACP extension headers consumed at interception and never sent upstream."""

LineageTransition = Literal["root", "spawn", "compact"]
LineageRequestKind = Literal["turn", "compaction"]
SessionStatus = Literal["running", "completed", "failed", "cancelled"]
CompactionStatus = Literal["in_progress", "completed", "failed", "cancelled"]


class _StrictLineageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CallLineage(_StrictLineageModel):
    """Provenance copied from one intercepted model request."""

    request_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    session_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    parent_session_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)
    context_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    previous_context_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)
    transition: LineageTransition
    compaction_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)
    depth: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_transition(self) -> CallLineage:
        if self.transition == "root":
            if self.parent_session_id is not None:
                raise ValueError("a root context cannot have a parent session")
            if self.previous_context_id is not None:
                raise ValueError("a root context cannot have a previous context")
        elif self.transition == "spawn":
            if self.parent_session_id is None:
                raise ValueError("a spawned context requires a parent session")
            if self.previous_context_id is not None:
                raise ValueError("a spawned context cannot have a previous context")
        elif self.previous_context_id is None:
            raise ValueError("a compacted context requires a previous context")
        if self.transition == "compact" and self.compaction_id is None:
            raise ValueError("a compacted context requires a compaction id")
        return self


class LineageSession(_StrictLineageModel):
    session_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    parent_session_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)
    depth: int = Field(ge=0)
    initial_context_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    spawned_by_request_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)
    status: SessionStatus


class LineageContext(_StrictLineageModel):
    context_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    session_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    previous_context_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)
    transition: LineageTransition
    compaction_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)


class LineageCompaction(_StrictLineageModel):
    compaction_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    session_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    source_context_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    target_context_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    summary_request_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    status: CompactionStatus


class LineageRequest(_StrictLineageModel):
    request_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    session_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    context_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    kind: LineageRequestKind
    compaction_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)


def _unique(items: list, attribute: str, label: str) -> dict[str, object]:
    indexed: dict[str, object] = {}
    for item in items:
        key = getattr(item, attribute)
        if key in indexed:
            raise ValueError(f"duplicate lineage {label} id: {key!r}")
        indexed[key] = item
    return indexed


def _assert_acyclic(
    records: Mapping[str, object], parent_attribute: str, label: str
) -> None:
    for start in records:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise ValueError(f"lineage {label} cycle at {current!r}")
            seen.add(current)
            record = records[current]
            current = getattr(record, parent_attribute)


class LineageManifest(_StrictLineageModel):
    """The complete recursive-session snapshot published by the harness."""

    sessions: list[LineageSession]
    contexts: list[LineageContext]
    compactions: list[LineageCompaction]
    requests: list[LineageRequest]

    @model_validator(mode="after")
    def validate_references(self) -> LineageManifest:
        sessions = _unique(self.sessions, "session_id", "session")
        contexts = _unique(self.contexts, "context_id", "context")
        compactions = _unique(self.compactions, "compaction_id", "compaction")
        requests = _unique(self.requests, "request_id", "request")

        roots = [
            session for session in self.sessions if session.parent_session_id is None
        ]
        if len(roots) != 1:
            raise ValueError("ACP lineage must contain exactly one root session")

        for session in self.sessions:
            parent = session.parent_session_id
            if parent is None:
                if session.depth != 0:
                    raise ValueError(
                        f"root lineage session {session.session_id!r} must have depth 0"
                    )
                if session.spawned_by_request_id is not None:
                    raise ValueError(
                        f"root lineage session {session.session_id!r} cannot have a spawn request"
                    )
            else:
                parent_session = sessions.get(parent)
                if parent_session is None:
                    raise ValueError(
                        f"lineage session {session.session_id!r} references unknown parent {parent!r}"
                    )
                if session.depth != parent_session.depth + 1:  # type: ignore[attr-defined]
                    raise ValueError(
                        f"lineage session {session.session_id!r} has inconsistent depth"
                    )
                if session.spawned_by_request_id is None:
                    raise ValueError(
                        f"lineage session {session.session_id!r} requires a spawn request"
                    )
            initial = contexts.get(session.initial_context_id)
            if (
                initial is None
                or initial.session_id != session.session_id  # type: ignore[attr-defined]
                or initial.previous_context_id is not None  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    f"lineage session {session.session_id!r} has an invalid initial context"
                )
        _assert_acyclic(sessions, "parent_session_id", "session")

        for context in self.contexts:
            session = sessions.get(context.session_id)
            if session is None:
                raise ValueError(
                    f"lineage context {context.context_id!r} references unknown session"
                )
            previous = context.previous_context_id
            if previous is None:
                expected = "root" if session.parent_session_id is None else "spawn"  # type: ignore[attr-defined]
                if context.transition != expected:
                    raise ValueError(
                        f"initial context {context.context_id!r} must transition as {expected!r}"
                    )
                if context.compaction_id is not None:
                    raise ValueError(
                        f"initial context {context.context_id!r} cannot have a compaction id"
                    )
            else:
                prior = contexts.get(previous)
                if prior is None or prior.session_id != context.session_id:  # type: ignore[attr-defined]
                    raise ValueError(
                        f"lineage context {context.context_id!r} has an invalid previous context"
                    )
                if context.transition != "compact" or context.compaction_id is None:
                    raise ValueError(
                        f"replacement context {context.context_id!r} must name its compaction"
                    )
                compaction = compactions.get(context.compaction_id)
                if (
                    compaction is None
                    or compaction.target_context_id != context.context_id  # type: ignore[attr-defined]
                ):
                    raise ValueError(
                        f"replacement context {context.context_id!r} has an invalid compaction"
                    )
        _assert_acyclic(contexts, "previous_context_id", "context")

        for session in self.sessions:
            initial_contexts = [
                context.context_id
                for context in self.contexts
                if context.session_id == session.session_id
                and context.previous_context_id is None
            ]
            if initial_contexts != [session.initial_context_id]:
                raise ValueError(
                    f"lineage session {session.session_id!r} must have exactly one initial context"
                )

        for request in self.requests:
            context = contexts.get(request.context_id)
            if request.session_id not in sessions or (
                context is None or context.session_id != request.session_id  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    f"lineage request {request.request_id!r} has an invalid session/context"
                )
            if request.kind == "compaction" and request.compaction_id is None:
                raise ValueError(
                    f"compaction request {request.request_id!r} requires a compaction id"
                )
            if request.compaction_id is not None:
                compaction = compactions.get(request.compaction_id)
                if compaction is None or compaction.session_id != request.session_id:  # type: ignore[attr-defined]
                    raise ValueError(
                        f"lineage request {request.request_id!r} has an invalid compaction"
                    )
                expected_context = (
                    compaction.source_context_id  # type: ignore[attr-defined]
                    if request.kind == "compaction"
                    else compaction.target_context_id  # type: ignore[attr-defined]
                )
                if request.context_id != expected_context:
                    raise ValueError(
                        f"lineage request {request.request_id!r} is on the wrong compaction context"
                    )
            elif request.kind == "turn" and context.compaction_id is not None:  # type: ignore[attr-defined]
                raise ValueError(
                    f"lineage request {request.request_id!r} is missing its context compaction"
                )

        for session in self.sessions:
            spawn_id = session.spawned_by_request_id
            if spawn_id is None:
                continue
            spawn = requests.get(spawn_id)
            if (
                spawn is None
                or spawn.kind != "turn"  # type: ignore[attr-defined]
                or spawn.session_id != session.parent_session_id  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    f"lineage session {session.session_id!r} has an invalid spawn request"
                )

        for compaction in self.compactions:
            source = contexts.get(compaction.source_context_id)
            target = contexts.get(compaction.target_context_id)
            request = requests.get(compaction.summary_request_id)
            if (
                compaction.session_id not in sessions
                or source is None
                or source.session_id != compaction.session_id  # type: ignore[attr-defined]
                or target is None
                or target.session_id != compaction.session_id  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    f"lineage compaction {compaction.compaction_id!r} has invalid contexts"
                )
            if (
                target.previous_context_id != compaction.source_context_id  # type: ignore[attr-defined]
                or target.transition != "compact"  # type: ignore[attr-defined]
                or target.compaction_id != compaction.compaction_id  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    f"lineage compaction {compaction.compaction_id!r} does not describe its target context"
                )
            if (
                request is None
                or request.kind != "compaction"  # type: ignore[attr-defined]
                or request.session_id != compaction.session_id  # type: ignore[attr-defined]
                or request.context_id != compaction.source_context_id  # type: ignore[attr-defined]
                or request.compaction_id != compaction.compaction_id  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    f"lineage compaction {compaction.compaction_id!r} has an invalid request"
                )
        return self


def extract_call_lineage(
    headers: Mapping[str, str],
) -> tuple[CallLineage | None, dict[str, str]]:
    """Parse and remove private ACP lineage headers from a provider-bound request.

    A partial lineage envelope is rejected: silently accepting it would turn an exact
    provenance channel into a heuristic one.  Ordinary requests with none of the private
    headers remain backward-compatible and carry no lineage.
    """

    forwarded = {
        name: value
        for name, value in headers.items()
        if name.lower() not in ACP_LINEAGE_HEADERS
    }
    normalized = {name.lower(): value for name, value in headers.items()}
    values = {
        name: value for name, value in normalized.items() if name in ACP_LINEAGE_HEADERS
    }
    if not values:
        return None, forwarded

    def get(name: str) -> str | None:
        value = values.get(name.lower())
        return value if value not in (None, "") else None

    required = (
        ACP_REQUEST_ID_HEADER,
        ACP_SESSION_ID_HEADER,
        ACP_CONTEXT_ID_HEADER,
        ACP_TRANSITION_HEADER,
        ACP_DEPTH_HEADER,
    )
    missing = [name for name in required if get(name) is None]
    if missing:
        raise ValueError(
            "incomplete ACP lineage headers: missing " + ", ".join(missing)
        )
    request_id = get(ACP_REQUEST_ID_HEADER)
    idempotency_key = normalized.get(IDEMPOTENCY_KEY_HEADER.lower())
    if idempotency_key is None:
        raise ValueError("ACP lineage requires Idempotency-Key")
    if idempotency_key != request_id:
        raise ValueError("ACP lineage request id must match Idempotency-Key")
    try:
        depth = int(get(ACP_DEPTH_HEADER) or "")
    except ValueError as error:
        raise ValueError(
            f"{ACP_DEPTH_HEADER} must be a non-negative integer"
        ) from error
    return (
        CallLineage(
            request_id=request_id,
            session_id=get(ACP_SESSION_ID_HEADER),
            parent_session_id=get(ACP_PARENT_SESSION_ID_HEADER),
            context_id=get(ACP_CONTEXT_ID_HEADER),
            previous_context_id=get(ACP_PREVIOUS_CONTEXT_ID_HEADER),
            transition=get(ACP_TRANSITION_HEADER),
            compaction_id=get(ACP_COMPACTION_ID_HEADER),
            depth=depth,
        ),
        forwarded,
    )
