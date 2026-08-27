"""Optional ACP lineage carried beside the training message graph.

The message DAG remains the source of training branches. This module describes the
runtime provenance that explains which recursive session and context epoch produced each
model call. ACP agents send the call-local part in private HTTP headers and publish the
full manifest through response ``_meta``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

LINEAGE_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"

ACP_LINEAGE_METADATA_KEY = "ai.prime.acp/lineage-v1"
ACP_REQUEST_ID_HEADER = "X-ACP-Lineage-Request-ID"

ACP_LINEAGE_HEADERS = frozenset({ACP_REQUEST_ID_HEADER.lower()})
"""Private ACP extension headers consumed at interception and never sent upstream."""

LineageTransition = Literal["root", "spawn", "compact"]
LineageRequestKind = Literal["turn", "compaction"]
SessionStatus = Literal["running", "completed", "failed", "cancelled"]
CompactionStatus = Literal["in_progress", "completed", "failed", "cancelled"]


class _StrictLineageModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


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
    """One compaction attempt; only completion materializes its target context."""

    compaction_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    session_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    source_context_id: str = Field(pattern=LINEAGE_ID_PATTERN)
    target_context_id: str | None = Field(default=None, pattern=LINEAGE_ID_PATTERN)
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
                    or compaction.status != "completed"  # type: ignore[attr-defined]
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
            request = requests.get(compaction.summary_request_id)
            if (
                compaction.session_id not in sessions
                or source is None
                or source.session_id != compaction.session_id  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    f"lineage compaction {compaction.compaction_id!r} has an invalid source context"
                )
            target = (
                contexts.get(compaction.target_context_id)
                if compaction.target_context_id is not None
                else None
            )
            if compaction.status == "completed" and (
                target is None
                or target.session_id != compaction.session_id
                or target.previous_context_id != compaction.source_context_id
                or target.transition != "compact"
                or target.compaction_id != compaction.compaction_id
            ):
                raise ValueError(
                    f"lineage compaction {compaction.compaction_id!r} does not describe its target context"
                )
            if (
                compaction.status != "completed"
                and compaction.target_context_id is not None
            ):
                raise ValueError(
                    f"non-completed lineage compaction {compaction.compaction_id!r} cannot have a target context"
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


def extract_lineage_request_id(
    headers: Mapping[str, str],
) -> tuple[str | None, dict[str, str]]:
    """Parse and remove the ACP lineage correlation ID from a provider request.

    The full execution graph arrives independently in ACP metadata. Ordinary requests
    without the private correlation header remain unchanged.
    """

    forwarded = {
        name: value
        for name, value in headers.items()
        if name.lower() not in ACP_LINEAGE_HEADERS
    }
    normalized = {name.lower(): value for name, value in headers.items()}
    request_id = normalized.get(ACP_REQUEST_ID_HEADER.lower())
    if request_id is None:
        return None, forwarded
    if re.fullmatch(LINEAGE_ID_PATTERN, request_id) is None:
        raise ValueError(f"{ACP_REQUEST_ID_HEADER} is not a valid lineage ID")
    return request_id, forwarded
