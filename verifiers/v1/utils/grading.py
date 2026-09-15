"""Grading-transport policy on top of sandbox artifact collection.

`collect` / `restore` in `artifacts.py` move tars. This module decides when a
declared path must exist so a verifier box is not scored against a stale state.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from verifiers.v1.utils.artifacts import ARTIFACTS_DIR, Artifact, collect

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime


class GradingCollect(StrEnum):
    """Whether a rollout tars declared artifacts into `trace.state` for a later box.

    `OFF` skips grading transport. `STRICT` fails if a declared (non-convention)
    path is missing. `BEST_EFFORT` records misses as `None` — Harbor's collection
    contract.
    """

    OFF = "off"
    STRICT = "strict"
    BEST_EFFORT = "best_effort"


MAX_BYTES = 32 * 1024 * 1024
"""Ceiling per grading collection. Sized for a delta, not a tree: the grading box
boots from the agent's image, so the repo is already there and only its output
has to travel."""


def _resolved_source(runtime: Runtime, source: str) -> str:
    workdir = PurePosixPath(getattr(runtime.config, "workdir", "") or "/")
    return str(workdir / source)


async def collect_strict(
    runtime: Runtime, artifacts: list[Artifact] | None = None
) -> dict[str, bytes | None]:
    """Collect like `collect`, then fail if a declared (non-convention) path is missing.

    A declared source that is missing was declared because grading needs it, and
    grading a partial state scores the rollout wrong rather than failing it. The
    implicit convention sweep is exempt — most tasks never write there.
    """
    collected = await collect(runtime, artifacts, max_bytes=MAX_BYTES)
    convention = PurePosixPath(ARTIFACTS_DIR)
    for artifact in artifacts or []:
        source = _resolved_source(runtime, artifact.source)
        if PurePosixPath(source) == convention:
            continue
        if collected.get(source) is None:
            raise RuntimeError(
                f"declared artifact {source!r} does not exist in the runtime"
            )
    return collected


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
            return await collect_strict(runtime, artifacts)
        case GradingCollect.BEST_EFFORT:
            return await collect(runtime, artifacts, max_bytes=MAX_BYTES)
