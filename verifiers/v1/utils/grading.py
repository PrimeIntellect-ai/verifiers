"""Grading-transport policy on top of sandbox artifact collection.

`collect` / `restore` in `artifacts.py` move tars. This module decides when a
declared path must exist so a verifier box is not scored against a stale state.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from verifiers.v1.utils.artifacts import Artifact, collect

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime


class GradingCollect(StrEnum):
    """Whether a rollout tars declared artifacts into `trace.state` for a later box.

    `OFF` skips grading transport. `STRICT` fails if a declared (non-convention)
    path is missing. `BEST_EFFORT` records misses as `None` — Harbor's collection
    contract. Both still raise when the 32MB transport cap is exceeded.
    """

    OFF = "off"
    STRICT = "strict"
    BEST_EFFORT = "best_effort"


MAX_BYTES = 32 * 1024 * 1024
"""Ceiling per grading collection. Sized for a delta, not a tree: the grading box
boots from the agent's image, so the repo is already there and only its output
has to travel."""


async def grading_collect(
    runtime: Runtime,
    artifacts: list[Artifact] | None,
    grading_collect: GradingCollect,
) -> dict[str, bytes | None] | None:
    """Tar declared artifacts per `grading_collect`. `OFF` returns None."""
    match grading_collect:
        case GradingCollect.OFF:
            return None
        case GradingCollect.STRICT:
            return await collect(
                runtime,
                artifacts,
                max_bytes=MAX_BYTES,
                missing="raise",
                on_limit="raise",
            )
        case GradingCollect.BEST_EFFORT:
            return await collect(
                runtime,
                artifacts,
                max_bytes=MAX_BYTES,
                missing="omit",
                on_limit="raise",
            )
