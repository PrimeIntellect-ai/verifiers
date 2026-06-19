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
    ignore_dockerfile: bool = True
    """Keep harbor from rejecting the Dockerfile-only tasks; `load_tasks` sets the real image."""
    use_prime_registry: bool = False
    """Resolve task images against prime's Artifact Registry (`REGISTRY_PREFIX`) instead of the
    dataset's public Docker Hub `swebench/sweb.eval.*` image. Only works on runtimes with GCP
    pull credentials (e.g. prime training); the default public image works anywhere. Mirrors
    `scaleswe-v1` / `r2e-gym-v1`."""


def from_image(task_dir: Path) -> str:
    """The image a task's Dockerfile builds on (`FROM swebench/sweb.eval.*`)."""
    for line in (task_dir / "environment" / "Dockerfile").read_text().splitlines():
        if line.strip().upper().startswith("FROM "):
            return line.split(None, 1)[1].strip()
    raise ValueError(f"{task_dir.name}: no FROM in environment/Dockerfile")


class SWEBenchVerifiedTaskset(
    HarborTaskset, vf.Taskset[HarborTask, SWEBenchVerifiedConfig]
):
    def load_tasks(self) -> list[HarborTask]:
        # TODO: once we have persistent public image caches, build each task's Dockerfile
        # (FROM swebench/sweb.eval.*) dynamically and cache the result — then we won't need to
        # resolve the prebuilt image here at all.
        tasks = []
        for task in super().load_tasks():
            base = from_image(Path(task.task_dir))
            # The mirror keeps the source repo path (`swebench/sweb.eval.*`), matching the
            # composable swebench taskset's `{REGISTRY}/{instance_image_key}`.
            image = (
                f"{REGISTRY_PREFIX}/{base}" if self.config.use_prime_registry else base
            )
            # SWE-bench instance images check out the repo at /testbed (the image's WORKDIR);
            # without pinning it the runtime's default workdir drops the agent in an empty dir.
            tasks.append(
                task.model_copy(update={"image": image, "workdir": "/testbed"})
            )
        return tasks
