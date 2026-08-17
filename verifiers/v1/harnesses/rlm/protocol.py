"""RLM extensions at ACP and model-interception boundaries."""

import re
from collections.abc import Mapping
from typing import Any

from verifiers.v1.logical_calls import LogicalCall, ResolvedCallHeaders

RLM_LINEAGE_VERSION_HEADER = "x-rlm-lineage-version"
RLM_LINEAGE_METADATA_KEY = "ai.prime.rlm/lineage-v1"
RLM_LINEAGE_HEADERS = {
    "session_id": "x-rlm-session-id",
    "invocation_id": "x-rlm-invocation-id",
    "parent_invocation_id": "x-rlm-parent-invocation-id",
    "segment_id": "x-rlm-segment-id",
    "call_id": "x-rlm-call-id",
    "parent_call_id": "x-rlm-parent-call-id",
    "depth": "x-rlm-depth",
    "call_kind": "x-rlm-call-kind",
}
_LINEAGE_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class RLMLineageResolver:
    def resolve(
        self, headers: Mapping[str, str], *, session_id: str
    ) -> ResolvedCallHeaders:
        forwarded = {
            name: value
            for name, value in headers.items()
            if not name.lower().startswith("x-rlm-")
        }
        version = headers.get(RLM_LINEAGE_VERSION_HEADER)
        if version is None:
            return ResolvedCallHeaders(call=None, forward_headers=forwarded)
        if version != "1":
            raise ValueError(f"unsupported RLM lineage version {version!r}")
        values = {
            name: headers.get(header) for name, header in RLM_LINEAGE_HEADERS.items()
        }
        required = (
            "session_id",
            "invocation_id",
            "segment_id",
            "call_id",
            "depth",
            "call_kind",
        )
        missing = [name for name in required if values[name] is None]
        if missing:
            raise ValueError(f"RLM lineage is missing fields: {missing}")
        for name in (
            "session_id",
            "invocation_id",
            "parent_invocation_id",
            "segment_id",
            "call_id",
            "parent_call_id",
        ):
            value = values[name]
            if value is not None and _LINEAGE_ID_RE.fullmatch(value) is None:
                raise ValueError(f"RLM lineage field {name!r} is invalid")
        if values["session_id"] != session_id:
            raise ValueError("RLM lineage session ID does not match the rollout")
        try:
            depth = int(values["depth"])
        except (TypeError, ValueError) as error:
            raise ValueError("RLM lineage depth is invalid") from error
        if depth < 0:
            raise ValueError("RLM lineage depth is invalid")
        if values["call_kind"] not in ("turn", "compaction"):
            raise ValueError("RLM lineage call kind is invalid")
        lineage: dict[str, Any] = {"version": 1, **values, "depth": depth}
        return ResolvedCallHeaders(
            call=LogicalCall(
                key=values["call_id"],
                metadata={RLM_LINEAGE_METADATA_KEY: lineage},
            ),
            forward_headers=forwarded,
        )


RLM_LINEAGE_RESOLVER = RLMLineageResolver()
