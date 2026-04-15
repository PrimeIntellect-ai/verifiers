from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from verifiers.types import (
    EpisodeResult,
    EpisodeStart,
    TurnReq,
    TurnResp,
)


@runtime_checkable
class MultiActorEnv(Protocol):
    """Protocol for environments that expose explicit multi-actor episode sessions."""

    def start_episode(
        self,
        example: dict[str, Any],
        sample_index: int,
    ) -> EpisodeStart: ...

    async def submit_ready_turns(
        self,
        responses: list[TurnResp],
    ) -> list[TurnReq]: ...

    async def finalize_episode(
        self,
        episode_id: str,
    ) -> EpisodeResult: ...
