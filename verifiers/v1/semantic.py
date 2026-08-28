"""Optional semantic edges carried beside the physical training-message graph.

ACP harnesses identify model requests before inference, while Verifiers message-node IDs
only exist after a response commits. The harness therefore publishes semantic edges over
opaque request IDs. Verifiers resolves those IDs to sampled assistant nodes and stores each
source as a semantic parent of the target node.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

MODEL_REQUEST_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
EDGE_TYPE_PATTERN = r"^[A-Za-z][A-Za-z0-9._:-]{0,127}$"

ACP_SEMANTIC_EDGES_METADATA_KEY = "ai.prime.acp/semantic-edges-v1"
ACP_MODEL_REQUEST_ID_HEADER = "X-ACP-Model-Request-ID"

ACP_EXTENSION_HEADERS = frozenset({ACP_MODEL_REQUEST_ID_HEADER.lower()})
"""Private ACP extension headers consumed at interception and never sent upstream."""


class _StrictSemanticModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SemanticEdge(_StrictSemanticModel):
    """A harness-declared relationship between two logical model requests.

    Request IDs are wire-level correlation handles. They let a harness describe the
    relationship before it can know Verifiers-local message-node indexes.
    """

    source_request_id: str = Field(pattern=MODEL_REQUEST_ID_PATTERN)
    target_request_id: str = Field(pattern=MODEL_REQUEST_ID_PATTERN)
    type: str = Field(pattern=EDGE_TYPE_PATTERN)

    @model_validator(mode="after")
    def reject_self_edge(self) -> SemanticEdge:
        if self.source_request_id == self.target_request_id:
            raise ValueError("semantic edge cannot link a request to itself")
        return self


class ParentLink(_StrictSemanticModel):
    """One semantic parent of a ``MessageNode``.

    ``node`` is an index into the containing ``Trace.nodes``. The child is the
    ``MessageNode`` carrying this link.
    """

    node: int = Field(ge=0)
    type: str = Field(pattern=EDGE_TYPE_PATTERN)


class SemanticEdgeSet(_StrictSemanticModel):
    """A harness-published set of semantic edges over logical request IDs.

    Edge labels are intentionally extensible. Initial harnesses use ``continuation``,
    ``compaction``, ``subagent_call``, and ``subagent_return``; consumers must preserve
    unknown labels.
    """

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


def extract_acp_model_request_id(
    headers: Mapping[str, str],
) -> tuple[str | None, dict[str, str]]:
    """Parse and remove the ACP model-request correlation ID.

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
    if re.fullmatch(MODEL_REQUEST_ID_PATTERN, request_id) is None:
        raise ValueError(
            f"{ACP_MODEL_REQUEST_ID_HEADER} is not a valid model request ID"
        )
    return request_id, forwarded
