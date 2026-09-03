"""ACP semantic edges carried beside the physical training-message graph."""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

ACP_REQUEST_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
EDGE_TYPE_PATTERN = r"^[A-Za-z][A-Za-z0-9._:-]{0,127}$"
"""Semantic label syntax, for example ``subagent_return`` or ``vendor:review``."""

ACP_SEMANTIC_EDGES_METADATA_KEY = "ai.prime.acp/semantic-edges-v1"
ACP_TRAINING_EXCLUSIONS_METADATA_KEY = "ai.prime.acp/training-exclusions-v1"
ACP_MODEL_REQUEST_ID_HEADER = "X-ACP-Model-Request-ID"

ACP_EXTENSION_HEADERS = frozenset({ACP_MODEL_REQUEST_ID_HEADER.lower()})
"""Private ACP extension headers consumed at interception and never sent upstream."""


class ACPInfo(BaseModel):
    """Metadata advertised by an ACP harness for one intercepted model request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str = Field(pattern=ACP_REQUEST_ID_PATTERN)


class SemanticEdge(BaseModel):
    """A harness-declared relationship between two logical model requests.

    Request IDs are wire-level correlation handles. They let a harness describe the
    relationship before it can know Verifiers-local message-node indexes.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    source_request_id: str = Field(pattern=ACP_REQUEST_ID_PATTERN)
    target_request_id: str = Field(pattern=ACP_REQUEST_ID_PATTERN)
    type: str = Field(pattern=EDGE_TYPE_PATTERN)

    @model_validator(mode="after")
    def reject_self_edge(self) -> SemanticEdge:
        if self.source_request_id == self.target_request_id:
            raise ValueError("semantic edge cannot link a request to itself")
        return self


class ParentLink(BaseModel):
    """One semantic parent of a ``MessageNode``.

    ``node`` is an index into the containing ``Trace.nodes``. The child is the
    ``MessageNode`` carrying this link.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    node: int = Field(ge=0)
    type: str = Field(pattern=EDGE_TYPE_PATTERN)


class SemanticEdgeSet(BaseModel):
    """A harness-published set of semantic edges over logical request IDs.

    Edge labels are intentionally extensible. Initial harnesses use ``continuation``,
    ``compaction_attempt``, ``compaction``, ``subagent_call``, and ``subagent_return``;
    consumers must preserve unknown labels.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    edges: list[SemanticEdge]

    @model_validator(mode="after")
    def validate_edges(self) -> SemanticEdgeSet:
        identities: set[tuple[str, str, str]] = set()
        children: dict[str, list[str]] = {}
        nodes: set[str] = set()
        for edge in self.edges:
            identity = (
                edge.source_request_id,
                edge.target_request_id,
                edge.type,
            )
            if identity in identities:
                raise ValueError(f"duplicate semantic edge: {identity!r}")
            identities.add(identity)
            children.setdefault(edge.source_request_id, []).append(
                edge.target_request_id
            )
            nodes.update((edge.source_request_id, edge.target_request_id))

        indegree = dict.fromkeys(nodes, 0)
        for targets in children.values():
            for target in targets:
                indegree[target] += 1

        stack = [node for node, degree in indegree.items() if degree == 0]
        visited = 0
        while stack:
            node = stack.pop()
            visited += 1
            for child in children.get(node, ()):
                indegree[child] -= 1
                if indegree[child] == 0:
                    stack.append(child)
        if visited != len(nodes):
            raise ValueError("semantic edge cycle detected")
        return self


class TrainingExclusionSet(BaseModel):
    """Logical model requests whose sampled tokens must remain out of training."""

    model_config = ConfigDict(extra="forbid", strict=True)

    request_ids: list[str]

    @model_validator(mode="after")
    def reject_duplicates(self) -> TrainingExclusionSet:
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("duplicate excluded model request ID")
        for request_id in self.request_ids:
            if re.fullmatch(ACP_REQUEST_ID_PATTERN, request_id) is None:
                raise ValueError(f"invalid excluded model request ID: {request_id!r}")
        return self


def extract_acp_info(
    headers: Mapping[str, str],
) -> tuple[ACPInfo | None, dict[str, str]]:
    """Parse ACP request metadata and remove its private transport header.

    The semantic edge set arrives independently in ACP metadata. Ordinary requests
    without the private correlation header remain unchanged.
    """

    forwarded = {
        name: value
        for name, value in headers.items()
        if name.lower() not in ACP_EXTENSION_HEADERS
    }
    normalized = {name.lower(): value for name, value in headers.items()}
    request_id = normalized.get(ACP_MODEL_REQUEST_ID_HEADER.lower())
    if request_id is None:
        return None, forwarded
    if re.fullmatch(ACP_REQUEST_ID_PATTERN, request_id) is None:
        raise ValueError(f"{ACP_MODEL_REQUEST_ID_HEADER} is not a valid ACP request ID")
    return ACPInfo(request_id=request_id), forwarded
