"""swebench-verified-v1 — SWE-bench Verified on harbor, using prime's prebuilt images.

A thin wrapper over `harbor-v1` pinned to the `swebench-verified` dataset. Those harbor tasks ship a
Dockerfile (`FROM swebench/sweb.eval.*`, the public SWE-bench instance image) rather than a pullable
`[environment].docker_image`, so harbor-v1 would reject them (it doesn't build Dockerfiles). Instead
we point each task at prime's prebuilt mirror of that same image, which the prime runtime pulls
directly — no build.

Needs the `harbor` CLI (`uv tool install harbor`) and a container runtime that can reach the registry
(prime).
"""

from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from tasksets.harbor_v1 import HarborConfig, HarborTask, HarborTaskset

# Prime's Artifact Registry mirror of the SWE-bench instance images.
REGISTRY_PREFIX = "us-central1-docker.pkg.dev/prime-intellect-platform/prod-sandbox"


class SWEBenchVerifiedConfig(HarborConfig):
    dataset: Literal["swebench-verified"] = "swebench-verified"
    use_harness_image: bool = True
    """Keep harbor from rejecting the Dockerfile-only tasks; `load_tasks` sets the real image."""


class SWEBenchVerifiedTaskset(
    HarborTaskset, vf.Taskset[HarborTask, SWEBenchVerifiedConfig]
):
    def load_tasks(self) -> list[HarborTask]:
        # TODO: once we have persistent public image caches, build each task's Dockerfile
        # (FROM swebench/sweb.eval.*) dynamically and cache the result — then we won't need to
        # rewrite the image to prime's prebuilt mirror here.
        return [
            task.model_copy(update={"image": _prime_image(Path(task.task_dir))})
            for task in super().load_tasks()
        ]


def _prime_image(task_dir: Path) -> str:
    """Map a task's Dockerfile `FROM swebench/sweb.eval.*` to prime's prebuilt mirror image."""
    for line in (task_dir / "environment" / "Dockerfile").read_text().splitlines():
        if line.strip().upper().startswith("FROM "):
            base = line.split(None, 1)[1].strip().rsplit("/", 1)[-1]
            return f"{REGISTRY_PREFIX}/{base}"
    raise ValueError(f"{task_dir.name}: no FROM in environment/Dockerfile")


__all__ = ["SWEBenchVerifiedTaskset"]
