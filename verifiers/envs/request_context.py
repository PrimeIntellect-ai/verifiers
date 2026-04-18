from __future__ import annotations

from dataclasses import dataclass

from verifiers.utils.usage_utils import StateUsageTracker


@dataclass(slots=True)
class ModelRequestContext:
    """Ephemeral per-request metadata for one model call.

    This carries control data that should influence how a single
    inference request is routed or accounted, but should not be treated
    as durable rollout state.
    """

    lineage_key: str | None = None
    usage_tracker: StateUsageTracker | None = None
