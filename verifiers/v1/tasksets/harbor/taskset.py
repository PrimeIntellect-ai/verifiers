"""Harbor tasksets backed by Harbor Hub packages.

The Harbor CLI downloads and caches each task directory. Its verifier runs in the
runtime the harness edited — or, when the task asks for it with
``[verifier].environment_mode = "separate"``, in a second box the agent never
touched, carrying only what the task declared — the rollout provisions and
grades that box. Either way the score lands in
``/logs/verifier/reward.json`` or the legacy ``reward.txt``.

A pullable ``[environment].docker_image`` becomes ``TaskData.image``. Verifiers does
not build Dockerfile-only environments, so those are rejected unless ``ignore_dockerfile``
deliberately uses the harness runtime image. Tasks without an environment also use that
image unless ``require_image`` is set. Separate verifiers prefer their own image
or tests build definition, then fall back to the agent environment. Unsupported verifier builds are rejected unless
``ignore_dockerfile`` explicitly selects the fallback image.
"""

import asyncio
import hashlib
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from verifiers.v1 import toolbundles
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.scoring import ScoringConfig
from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.errors import SandboxError, SubmissionRefused, TaskError
from verifiers.v1.runtimes import ProgramResult, Runtime, RuntimeConfig
from verifiers.v1.runtimes.compose import ComposeSpec
from verifiers.v1.runtimes.deployment import Deployment
from verifiers.v1.state import State
from verifiers.v1.task import Task, TaskData, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset
from verifiers.v1.tasksets.harbor.context_owners import restore_owners
from verifiers.v1.tasksets.harbor.criteria import (
    CriteriaJudgeConfig,
    TaskCriteria,
    TaskDecisionJudgeConfig,
    criteria_judge_config,
    grade_criteria,
    load_task_criteria,
)
from verifiers.v1.tasksets.harbor.entrypoint import restart_services, start_entrypoint
from verifiers.v1.toolbundles import ToolMount
from verifiers.v1.trace import Trace
from verifiers.v1.utils.artifacts import MAX_ARTIFACT_BYTES, Artifact, collect
from verifiers.v1.utils.decorators import reward
from verifiers.v1.utils.env import resolve_env
from verifiers.v1.validation import (
    ValidationResult,
    ValidationRunner,
    validate_pair,
)

logger = logging.getLogger(__name__)

CACHE = Path.home() / ".cache" / "harbor"
HARBOR_INSTALL_HINT = "uv sync --python 3.12 --extra harbor"
REWARD_JSON = "/logs/verifier/reward.json"
AGENT_LOGS = "/logs/agent"
TRAJECTORY = f"{AGENT_LOGS}/trajectory.json"
FINAL_REPLY = f"{AGENT_LOGS}/final_reply.md"
"""The solver's final reply, staged from the trace for graders (a judge criterion's
evidence) beside its transcript."""
GRADER_PATHS = ("/tests", "/logs/verifier", AGENT_LOGS)
"""Where a Harbor verifier keeps its tests, writes its reward and reads the agent's
transcript: grader-owned, so no restored submission may land or link in them."""
MAX_REWARD_BYTES = 1024 * 1024
VERIFIER_OUTPUT_BYTES = 16_000
"""How much of the verifier's output, and of each log it leaves, a trace keeps."""
MAX_VERIFIER_OUTPUT_CHARS = 16 * 1024
"""How much of the verifier's stdout and stderr survive a failed scoring attempt."""
VERIFIER_PROBE_SECONDS = 120
"""How long a verifier's box has to answer a trivial command after `test.sh` outlived its
timeout: one that answers graded a submission that ran too long, one that does not is
an infrastructure failure."""
# Booleans count as 1/0, as Harbor reads them (CyberGym's verifier writes flags alongside its reward).
REWARD_JSON_ADAPTER = TypeAdapter(
    dict[str, bool | float],
    config=ConfigDict(strict=True, allow_inf_nan=False),
)

REWARD_SCALAR_ADAPTER = TypeAdapter(
    float, config=ConfigDict(strict=True, allow_inf_nan=False)
)


class HarborTaskConfig(TaskConfig):
    reward_key: str = Field("reward", min_length=1)
    """Named verifier value used as the training reward; other values remain metrics."""
    compose_registry_mirrors: list[str] = Field(default_factory=list)
    """Docker registry mirrors on a Compose provider VM; empty keeps host defaults.
    Service images, pinned digests and build inputs remain authored."""
    mcp_servers: list[dict] = Field(default_factory=list)
    """Task-declared connections, bound from HarborData during construction."""
    criteria_judge: CriteriaJudgeConfig = TaskDecisionJudgeConfig()
    """The judge that answers a task's `judge` criteria (`tests/criteria.toml`, see
    `harbor.criteria`): `id = "decision"` (clef on Prime's evaluations API, the default;
    reads image evidence) or `id = "rubric"` (a chat model on Prime inference; text
    only)."""

    @field_validator("criteria_judge", mode="before")
    @classmethod
    def _criteria_judge(cls, value: object) -> object:
        return criteria_judge_config(value)


class HarborConfig(TasksetConfig):
    shared_grading: Literal["open", "declared"] = Field("open", exclude=True)
    """The network a shared-box verifier grades under. "open" (this branch's default): online,
    as before verifiers followed Harbor's verifier phase policy (#2909); vendor graders
    install their tools and reach their judges at grade time (SWE-Marathon, CyberStrike).
    "declared": Harbor's verifier phase policy (`shared_grading_network`). Left out of the
    config dump, so it never changes task identity."""
    artifact_max_bytes: int = Field(MAX_ARTIFACT_BYTES, gt=0)
    """Total archive bytes collected from one solver, across all its services."""
    task: HarborTaskConfig = HarborTaskConfig()
    dataset: str = "harbor/hello-world"
    """A Harbor Hub package id ("org/name" or "org/name@ref"), where ref is a
    tag, integer revision, or sha256 digest. Legacy registries selected with `repo`,
    `registry_path`, or `registry_url` use a bare dataset name ("name" or "name@version")."""
    repo: str | None = None
    """Optional Harbor `--repo` registry selector, e.g. "org/repo@ref"."""
    registry_path: Path | None = None
    """Optional Harbor `--registry-path` selector. Local unless `repo` is also set."""
    registry_url: str | None = None
    """Optional Harbor `--registry-url` selector for a raw registry.json URL."""
    tasks: list[str] | None = None
    """Optional subset of task names to load (None = all)."""
    ignore_timeouts: bool = True
    """Drop each task's declared agent timeout and scoring budget so rollouts use only the
    run-level `--timeout.*` limits (none by default). Task timeouts are authored
    against Harbor's runtime and confound model capability with inference speed; set
    False to apply them anyway. `test.sh` itself still runs under the task's declared
    `[verifier].timeout_sec`, as Harbor runs it (`HarborTask.verifier_timeout`): how
    long grading takes does not depend on the model's speed, and a submission that
    hangs its tests would otherwise hold the run for the whole scoring budget."""
    timeout_multiplier: float = Field(1.0, gt=0)
    """Scale each task's agent and verifier timeouts. Only applies with
    `ignore_timeouts=False`."""
    resource_multiplier: float = Field(1.0, gt=0)
    """Scale each task's CPU, memory, and disk requests. GPU requests are unchanged."""
    require_image: bool = False
    """For a task with NO declared environment at all (no docker_image, no Dockerfile),
    whether to reject it (True) or run it on the runtime's default image (False). A task
    whose environment is a `Dockerfile` is rejected too (building Dockerfiles isn't
    supported), unless `ignore_dockerfile`."""
    ignore_dockerfile: bool = False
    """Run a task whose environment is only a `Dockerfile` on the harness runtime's image
    instead of rejecting it. The Dockerfile is NOT built, so the task scores against the
    harness image rather than its declared environment — only correct when that image already
    has what the task needs (e.g. you've pointed the runtime at the right image)."""
    boots_kernels: bool = False
    """Whether this taskset boots the kernel a task declares in `[metadata.prime_kernel]` (a
    kernel feature the provider's guest kernel lacks, such as Landlock). A plain Harbor
    taskset does not, so it refuses such a task rather than running it on a box that cannot
    grade it; a kernel-booting taskset (sft-gen's `sft-gen-kernel-harbor`) sets it."""
    ignore_separate_verifier: bool = False
    """Grade every task in the agent's own box, even one whose `[verifier]` asks for a
    separate one. Trades the isolation for a sandbox per task; useful when provisioning
    is the bottleneck. Note what it gives up: the grader becomes reachable by the agent
    that just ran."""


class Author(BaseModel):
    name: str | None = None
    email: str | None = None


class CollectHook(BaseModel):
    """One `[[verifier.collect]]` command, run in its service by `finalize`."""

    command: str
    timeout_sec: float = 600.0
    service: str = "main"
    user: str | None = Field(None, exclude_if=lambda user: user is None)
    """Who runs it; None (or root) is the runtime's default user. Left out of the dump
    while unset, so tasks that name none keep the data (and call fingerprints) they had
    before the field existed."""


class HarborArtifact(Artifact):
    service: str = "main"
    """The Compose service to collect from; the grader restores at the same path."""


class VerifierConfig(BaseModel):
    """The box this task's verifier wants, when it wants one of its own.

    `None` on `HarborData` means shared — grade where the agent worked, which is still
    Harbor's default and every task that says nothing."""

    resources: TaskResources = TaskResources()
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    healthcheck: dict | None = None
    fresh_copy: bool = False
    """Whether this came from Harbor's fresh copy of `[environment]` rather than a
    declared `[verifier.environment]`. A fresh copy inherits the agent box's resolved
    resources; a declared environment states its own, and what it omits falls back to
    the run's rather than to the agent's task-derived values."""
    network: NetworkPolicyConfig = Field(default_factory=NetworkPolicyConfig)
    """What the verifier may reach, from the verifier's network mode. `allow=["*"]` is
    unrestricted; `allow=[]` is Harbor's `no-network` / `allow_internet = false`."""


class HarborData(TaskData):
    """Parsed ``task.toml`` metadata plus the host-side verifier directory.

    Base ``TaskData`` fields hold the prompt, resolved image, timeout, resources,
    name, and description. The remaining fields mirror Harbor metadata.
    """

    keywords: list[str] = Field(default_factory=list)
    authors: list[Author] = Field(default_factory=list)
    difficulty: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    task_dir: str = ""
    """Host path to the task dir; used to stage tests/ to verify."""
    upload_environment: bool = False
    """Whether to stage environment/ into the workdir."""
    env: dict[str, str] = Field(default_factory=dict)
    """Raw `[environment.env]` templates, resolved only when the runtime starts."""
    healthcheck: dict | None = None
    mcp_servers: list[dict] = Field(default_factory=list)
    """Task-declared MCP servers, preserved for served-task reconstruction. Includes
    one entry per tool bundle in `tools`."""
    tools: list[ToolMount] = Field(default_factory=list)
    """`[[tools]]` bundles (`verifiers.v1.toolbundles`): setup installs, seeds and
    starts their servers, and a collect hook dumps each one's end state."""
    compose_host_image: str | None = None
    """Provider VM image hosting a Compose task's Docker daemon, from `task.toml`'s
    `[metadata.prime_compose] host_image`. Docker is installed when the image lacks it,
    and `docker save` archives it ships in /opt/verifiers/compose-images/ load before
    the services start (`compose.prepare_host_image` makes one). None uses a stock
    image."""
    verifier_image: str | None = None
    """Pullable image for a separate verifier, containing the complete `/tests` suite.
    None keeps the solver image and stages the task package's tests."""
    verifier_env: dict[str, str] = Field(default_factory=dict)
    """Raw [verifier.env] entries (literals or `${VAR}`/`${VAR:-default}` templates).
    Resolved against the host environment at scoring time, like `harbor run` — so a
    verifier that needs judge API keys or configuration actually receives them."""
    artifacts: list[HarborArtifact] = Field(default_factory=list)
    collect: list[CollectHook] = Field(default_factory=list)
    """`[[verifier.collect]]` blocks: commands that snapshot runtime state into files
    after the agent stops, so the files can travel to a grading box as artifacts."""
    verifier: VerifierConfig | None = None
    """The verifier's own box, when `[verifier].environment_mode` asks for one. None
    grades in the agent's box."""
    verifier_user: str | None = Field(None, exclude_if=lambda user: user is None)
    """`[verifier].user`: who runs `test.sh`; None (or root) is the box's default user.
    Left out of the dump while unset, like `CollectHook.user`."""
    criteria: TaskCriteria | None = Field(
        None, exclude_if=lambda criteria: criteria is None
    )
    """`tests/criteria.toml`: the criteria the grader decides and folds into the reward
    after `test.sh` runs. None grades by the verifier's reward file alone. Left out of
    the dump while unset, like `verifier_user`."""


class HarborTask(Task[HarborData, State, HarborTaskConfig]):
    """Stage and run Harbor's verifier inside the task's live runtime."""

    RESTORE_PROTECTED = GRADER_PATHS
    verifier_staged: bool = False

    async def validate(
        self,
        runtime: Runtime | RuntimeConfig,
        *,
        baseline: bool = True,
        gold: bool = True,
        baseline_max: float = 0.0,
        gold_min: float = 1.0,
        gold_rewards: dict[str, float] | None = None,
        baseline_rewards: list[str] | None = None,
        gold_source: Path | None = None,
        gold_target_dir: str = "/solution",
        gold_command: list[str] | None = None,
        timeout: TimeoutConfig | None = None,
        retries: RetryConfig | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        run_attempt: ValidationRunner | None = None,
    ) -> ValidationResult:
        """Grade untouched and gold work in independent native deployments.

        Each reward must meet its inclusive bound. Callers may repeat this check
        concurrently; neither attempt reuses the caller's live workspace.
        """
        return await validate_pair(
            self,
            runtime,
            baseline=baseline,
            gold=gold,
            baseline_max=baseline_max,
            gold_min=gold_min,
            gold_rewards=gold_rewards,
            baseline_rewards=baseline_rewards,
            gold_source=gold_source or Path(self.data.task_dir) / "solution",
            gold_target_dir=gold_target_dir,
            gold_command=gold_command,
            timeout=timeout,
            retries=retries,
            on_trace=on_trace,
            run_attempt=run_attempt,
        )

    def __init__(self, data: HarborData, config: HarborTaskConfig | None = None):
        super().__init__(data, config)
        # Each reconstructed row gets its own connections without mutating worker config.
        self.config = self.config.model_copy(update={"mcp_servers": data.mcp_servers})

    @classmethod
    def toolsets(cls, config: HarborTaskConfig):
        from .toolset import HarborMCPConfig, HarborMCPToolset

        return super().toolsets(config) + [
            HarborMCPToolset(HarborMCPConfig(colocated=True, server=server))
            for server in config.mcp_servers
        ]

    def deployment_spec(self):
        from verifiers.v1.runtimes.compose import ComposeSpec

        authored = (
            Path(self.data.task_dir) / "environment/docker-compose.yaml"
        ).is_file()
        sidecars = {
            m.bundle: m.sidecar_service()
            for m in self.data.tools
            if m.placement == "sidecar"
        }
        if authored or sidecars:
            return ComposeSpec(
                self.data.task_dir,
                file="environment/docker-compose.yaml" if authored else None,
                host_image=self.data.compose_host_image,
                registry_mirrors=tuple(self.config.compose_registry_mirrors),
                image_override=self.data.image is not None,
                services=sidecars,
            )
        return super().deployment_spec()

    @property
    def scoring(self) -> ScoringConfig:
        # Keep tasksets that populate verifier_env compatible with the common policy.
        return self.data.scoring.model_copy(
            update={"env": {**self.data.verifier_env, **(self.data.scoring.env or {})}}
        )

    def grading_task(self, runtime: Runtime, *, scoring: ScoringConfig):
        # Custom Harbor verifiers may still pass verifier_env explicitly to run().
        # Give them the merged raw values, without resolving or mutating solver data.
        task = self.with_data(verifier_env=dict(scoring.env or {}), scoring=scoring)
        if scoring.mode == "shared":
            return task.with_data(verifier=None, verifier_image=None)
        if task.data.verifier is None:
            task = task.with_data(
                verifier=VerifierConfig(fresh_copy=True, env=task.data.env)
            )
        if task.data.verifier_image is None:
            task = task.with_data(
                image=runtime.config.image, workdir=runtime.config.workdir
            )
        return type(self)(verifier_box_data(task.data), self.config)

    async def finalize_services(
        self, trace: Trace, targets: dict[str, Runtime]
    ) -> None:
        declared = dict.fromkeys(
            entry.service for entry in (*self.data.collect, *self.data.artifacts)
        )
        if missing := declared.keys() - targets.keys():
            raise TaskError(f"Unknown Compose services: {sorted(missing)}")
        sidecars = {name: targets[name] for name in declared if name != "main"}
        if sidecars:
            await self.finalize(trace, targets["main"], sidecars)

    def runtime_env(self) -> dict[str, str]:
        return resolve_env(self.data.env)

    async def setup(
        self,
        runtime: Runtime,
        deployment: Deployment | None = None,
        trace: Trace | None = None,
    ) -> None:
        # Harbor's Docker start runs the image's ENTRYPOINT before anything else, in the
        # container's own PID namespace.
        if not isinstance(self.deployment_spec(), ComposeSpec):
            await runtime.backend.isolate_processes()
        guarded = await guard_onednn(runtime)
        if guarded and trace is not None:
            trace.info["onednn_guard"] = guarded
        if (dockerfile := self.entrypoint_dockerfile()) is not None:
            # Docker's COPY gives root what Prime's builder left with the context's owner.
            await restore_owners(runtime, dockerfile)
            started = await start_entrypoint(runtime, dockerfile)
            if started is not None and trace is not None:
                box = "verifier_" if dockerfile.parent.name == "tests" else ""
                trace.info[f"{box}entrypoint"] = started
        if deployment is not None:
            declared = {
                entry.service for entry in (*self.data.collect, *self.data.artifacts)
            }
            if missing := declared - deployment.target_backends.keys():
                raise TaskError(f"Unknown Compose services: {sorted(missing)}")
        # The verifier's host variables are resolved at scoring; checking them here makes
        # a missing one (a judge's API key) fail before the solve rather than after it.
        resolve_env(self.scoring.env or {})
        if self.data.upload_environment:
            await runtime.write(
                "/tmp/environment.tgz",
                make_tar(Path(self.data.task_dir) / "environment"),
            )
            result = await runtime.run(
                [
                    "sh",
                    "-c",
                    "tar --no-same-owner -xzf /tmp/environment.tgz && rm /tmp/environment.tgz",
                ],
                {},
            )
            if result.exit_code:
                raise RuntimeError(
                    f"environment setup failed (exit {result.exit_code}): "
                    f"{(result.stderr or result.stdout).strip()[-500:]}"
                )
        if self.data.tools:
            await self.mount_tools(runtime, deployment)
        if self.data.healthcheck is not None:
            await self.wait_for_health(runtime, self.data.healthcheck)

    def entrypoint_dockerfile(self) -> Path | None:
        """The Dockerfile whose ENTRYPOINT this box's image starts with, when nothing
        else runs it: a Compose service's runs under the VM's Docker, and a box on the
        harness image (no pullable image, or the environment uploaded) has none."""
        task_dir = Path(self.data.task_dir)
        if (
            not self.data.task_dir
            or self.data.image is None
            or self.data.upload_environment
            or isinstance(self.deployment_spec(), ComposeSpec)
        ):
            return None
        verifier_box = (
            self.data.verifier is not None
            and self.data.verifier_image is not None
            and self.data.image == self.data.verifier_image
        )
        return task_dir / ("tests" if verifier_box else "environment") / "Dockerfile"

    def _sidecars(
        self, deployment: Deployment | None
    ) -> list[tuple[Runtime, ToolMount]]:
        sidecars = [m for m in self.data.tools if m.placement == "sidecar"]
        if sidecars and deployment is None:
            raise TaskError("sidecar tool bundles need the task's Compose deployment")
        return [(deployment.target(m.service), m) for m in sidecars]

    async def mount_tools(
        self, runtime: Runtime, deployment: Deployment | None
    ) -> None:
        """Install each tool bundle where it runs (`toolbundles.mount`): the box-placed
        ones and every endpoint in the solver's box, each sidecar's in its service."""
        boxed = [m for m in self.data.tools if m.placement == "box"]
        for target, mount in self._sidecars(deployment):
            await toolbundles.mount(target, [mount])
        await toolbundles.mount(
            runtime, boxed, endpoints=self.data.tools, agent_user=self.data.agent_user
        )

    async def restore(
        self,
        runtime: Runtime,
        deployment: Deployment | None = None,
        trace: Trace | None = None,
    ) -> None:
        """A box from a checkpoint keeps its files but no process: start the tool
        servers again over the state they left on disk, in a PID namespace again,
        after the services the image's entrypoint had started (`restart_services`)."""
        if not isinstance(self.deployment_spec(), ComposeSpec):
            await runtime.backend.isolate_processes()
        if self.entrypoint_dockerfile() is not None:
            restarted = await restart_services(runtime)
            if restarted is not None and trace is not None:
                trace.info["entrypoint_restart"] = restarted
        await toolbundles.start(
            runtime, [m for m in self.data.tools if m.placement == "box"]
        )
        for target, mount in self._sidecars(deployment):
            await toolbundles.start(target, [mount])

    async def wait_for_health(self, runtime: Runtime, healthcheck: dict) -> None:
        from harbor.environments.base import HealthcheckError
        from harbor.models.task.config import HealthcheckConfig

        healthcheck = HealthcheckConfig.model_validate(healthcheck)
        start_period_end = time.monotonic() + healthcheck.start_period_sec
        failures = 0
        while True:
            in_start_period = time.monotonic() < start_period_end
            try:
                async with asyncio.timeout(healthcheck.timeout_sec):
                    if (
                        await runtime.run(["sh", "-c", healthcheck.command], {})
                    ).exit_code == 0:
                        return
            except TimeoutError:
                pass

            if not in_start_period:
                failures += 1
                if failures >= healthcheck.retries:
                    raise HealthcheckError(
                        f"Healthcheck failed after {healthcheck.retries} consecutive "
                        f"retries: {healthcheck.command}"
                    )
            await asyncio.sleep(
                healthcheck.start_interval_sec
                if in_start_period
                else healthcheck.interval_sec
            )

    async def finalize(
        self,
        trace: Trace,
        runtime: Runtime,
        sidecars: dict[str, Runtime] | None = None,
    ) -> None:
        """Run collect hooks in authored order, then add artifacts to the trace, for
        main or, when given, the named sidecars.

        Harbor runs main's after the agent phase, which is exactly what `finalize`
        means. The rollout collects sidecars once main has stopped.

        Strict, unlike `harbor run`, which logs a failed hook and carries on: there the
        output is observability, here it is a grading input, and a silently absent file
        makes the verifier score a stale state instead of failing loudly.
        """
        runtimes = sidecars or {"main": runtime}
        for hook in [hook for hook in self.data.collect if hook.service in runtimes]:
            target = runtimes[hook.service]
            if hook.user not in (None, "root", "0"):
                target = target.with_user(hook.user)
            result = await self.run_collect_hook(target, hook)
            if not result.exit_code:
                continue
            detail = (result.stderr or result.stdout).strip()[-500:]
            if 0 < result.exit_code < 126:
                # The hook ran and its check failed. A verdict on the submission holds
                # when the same check runs again; a fault of the moment (a file changed
                # while a hook's tar read it) does not, and stays a retryable failure.
                again = await self.run_collect_hook(target, hook)
                if again.exit_code == result.exit_code:
                    raise SubmissionRefused(
                        f"collect hook refused the submission (exit {result.exit_code}, "
                        f"twice): {hook.command}\n{detail}"
                    )
                if not again.exit_code:
                    continue
            raise RuntimeError(
                f"collect hook failed (exit {result.exit_code}): "
                f"{hook.command}\n{detail}"
            )
        for service, source_runtime in runtimes.items():
            used = sum(len(data or b"") for data in trace.state.artifacts.values())
            collected = await collect(
                source_runtime,
                [
                    artifact
                    for artifact in self.data.artifacts
                    if artifact.service == service
                ],
                max_bytes=self.data.artifact_max_bytes - used,
                sweep=service == "main",
            )
            # Every service restores into the grader's one filesystem.
            roots = [PurePosixPath(root) for root in trace.state.artifacts]
            for source in map(PurePosixPath, collected):
                if any(
                    source.is_relative_to(root) or root.is_relative_to(source)
                    for root in roots
                ):
                    raise RuntimeError(
                        f"artifact {str(source)!r} overlaps another service's"
                    )
            trace.state.artifacts.update(collected)

    @staticmethod
    async def run_collect_hook(target: Runtime, hook):
        """One run of a collect hook; its own exit status, or a `RuntimeError` when it
        outlives its timeout (a stuck hook says nothing about the submission)."""
        try:
            return await asyncio.wait_for(
                target.run(["sh", "-c", hook.command], {}),
                hook.timeout_sec,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"collect hook timed out after {hook.timeout_sec}s: {hook.command}"
            ) from exc

    def record_refusal(self, trace: Trace) -> None:
        """The verifier's reward key scores 0, as `solved` records it."""
        trace.record_reward(self.config.reward_key, 0.0, 1.0)

    async def stage_verifier(self, trace: Trace, runtime: Runtime) -> None:
        if self.scoring.mode == "isolated" and (
            owned := grader_owned(trace.state.artifacts)
        ):
            raise TaskError(f"Harbor artifacts cannot restore into {owned}")
        if self.scoring.mode == "isolated":
            await self.own_restored_artifacts(trace, runtime)
        await self.stage_tests(runtime, wipe=self.data.verifier is not None)
        self.verifier_staged = True

    async def own_restored_artifacts(self, trace: Trace, runtime: Runtime) -> None:
        """Leave the grader's restored artifacts as Harbor's upload into a separate
        verifier does: the contents root-owned (`docker compose cp` keeps modes, not
        owners), and each restored directory, or a restored file's parent, world-writable.
        It matters to a verifier that runs as its own `[verifier].user`: it may write
        beside the submission, and the solver's uids, which `restore` carries over, mean
        nothing in the grader's image."""
        roots = [root for root, archive in trace.state.artifacts.items() if archive]
        if not roots:
            return
        command = (
            'for r; do if [ -d "$r" ] && [ ! -L "$r" ]; then '
            'chown -hR 0:0 "$r" && chmod 777 "$r"; '
            'else chown -h 0:0 "$r" && d=$(dirname "$r") && '
            '{ [ "$d" = / ] || chmod 777 "$d"; }; fi || exit 1; done'
        )
        result = await runtime.run(["sh", "-c", command, "vf-artifacts", *roots], {})
        if result.exit_code:
            raise TaskError(
                f"owning restored artifacts failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )

    async def stage_tests(self, runtime: Runtime, wipe: bool = False) -> None:
        """Use a dedicated verifier image's tests, or stage the task package's tests.

        Raises rather than scoring stale state: a leftover reward file — planted by
        the agent or shipped in the image — must be gone before `test.sh` runs, so a
        removal that fails must not fall through to reading it.
        """
        # Harbor's dedicated verifier image owns the complete test suite and its
        # dependencies. Mixing it with packaged tests can retain obsolete helpers.
        stage = "test -f /tests/test.sh"
        if self.data.verifier is None or self.data.verifier_image is None:
            await runtime.write(
                "/tmp/tests.tgz", make_tar(Path(self.data.task_dir) / "tests")
            )
            stage = (
                f"{'rm -rf /tests && ' if wipe else ''}"
                "mkdir -p /tests && tar -xzf /tmp/tests.tgz -C /tests"
            )
        # World-writable, as Harbor makes it: a verifier may log as an unprivileged
        # service user (`su postgres -c "pg_ctl -l /logs/verifier/pg.log"`).
        command = (
            "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt && "
            f"mkdir -p /logs/verifier && chmod 777 /logs/verifier && {stage}"
        )
        result = await runtime.run(["sh", "-c", command], {})
        if result.exit_code:
            raise TaskError(
                f"staging tests failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )

    @reward(weight=1.0)
    async def solved(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        if self.data.verifier is not None:
            if not self.verifier_staged:
                raise TaskError(
                    f"task {self.data.name!r} declares a separate verifier "
                    '([verifier].environment_mode = "separate"); grade it through '
                    "the standard task execution lifecycle, or force shared "
                    "grading with --taskset.ignore-separate-verifier"
                )
        else:
            # Grading is trusted: unless scoring declares its network, a shared-box
            # verifier must not inherit the agent phase's network restrictions (Harbor
            # test scripts install their tools).
            if self.scoring.network is None:
                await runtime.set_network(NetworkPolicyConfig())
            if not self.verifier_staged:
                await self.stage_tests(runtime)
        return await self.run_verifier(runtime, trace)

    async def run_verifier(
        self, runtime: Runtime, trace: Trace
    ) -> float | dict[str, float]:
        await self.stage_trajectory(runtime, trace)
        # The reward file is authoritative even when the script exits nonzero.
        timeout = self.verifier_timeout()
        try:
            async with asyncio.timeout(timeout) as limit:
                result = await self.launch_verifier(runtime, trace)
        except TimeoutError:
            if not limit.expired():
                raise
            return await self.verifier_timed_out(runtime, trace, timeout)
        await self.record_verifier_output(runtime, trace, result)
        scores = await self.read_reward_json(runtime)
        if self.data.criteria is not None:
            scores = await self.grade_criteria(runtime, trace, scores)
        if scores is None:
            try:
                value = (
                    (
                        await runtime.read(
                            "/logs/verifier/reward.txt", max_bytes=MAX_REWARD_BYTES
                        )
                    )
                    .decode()
                    .strip()
                )
                scores = {"reward": REWARD_SCALAR_ADAPTER.validate_python(float(value))}
            except (SandboxError, OSError, ValueError) as exc:
                # The trace of a failed grading may not survive it: the error carries
                # the logs the verifier left (a service's own log names the cause).
                logs = log_tails(trace.info.get("verifier", {}).get("files", ""))
                raise TaskError(
                    "Harbor verifier produced no reward.json and no valid reward.txt "
                    f"(exit {result.exit_code}): "
                    f"{(result.stderr or result.stdout).strip()[-500:]}"
                    + (f"\nverifier logs:\n{logs[-8000:]}" if logs else "")
                ) from exc
        key = self.config.reward_key
        if key not in scores:
            raise TaskError(
                f"Harbor verifier reward key {key!r} is missing; "
                f"available keys: {sorted(scores)}. "
                "Set task.reward_key to select a named verifier reward."
            )
        trace.record_metrics(
            {name: value for name, value in scores.items() if name != key}
        )
        return {key: scores[key]}

    def verifier_timeout(self) -> float | None:
        """How long `test.sh` may run: the task's declared `[verifier].timeout_sec`, or
        its scoring budget when that is longer (a `timeout_multiplier` scales it). None
        when the task declares neither. Read from the task's `task.toml` rather than
        kept in the task data, so the task keeps its identity and finished calls."""
        declared = declared_verifier_timeout(self.data.task_dir)
        budget = self.data.timeout.scoring
        if declared is None or budget is None:
            return declared if budget is None else budget
        return max(declared, budget)

    async def verifier_timed_out(
        self, runtime: Runtime, trace: Trace, timeout: float
    ) -> dict[str, float]:
        """`test.sh` outlived its timeout: graded 0, as Harbor fails the trial
        (`VerifierTimeoutError`, never retried). If the box no longer answers, the hang
        was the box's, not the submission's: a retryable `SandboxError` instead."""
        try:
            async with asyncio.timeout(VERIFIER_PROBE_SECONDS):
                await runtime.run(["true"], {})
        except (SandboxError, TimeoutError) as exc:
            raise SandboxError(
                f"Harbor verifier ran past its {timeout:g}s timeout and its box does "
                f"not answer: {exc or type(exc).__name__}",
                kind="io",
            ) from exc
        logger.warning(
            "%s: verifier ran past its %gs timeout; graded 0", self.data.name, timeout
        )
        result = ProgramResult(
            exit_code=124, stdout="", stderr=f"verifier timed out after {timeout:g}s"
        )
        await self.record_verifier_output(runtime, trace, result)
        trace.info["verifier"]["timed_out"] = timeout
        return {self.config.reward_key: 0.0}

    async def grade_criteria(
        self, runtime: Runtime, trace: Trace, scores: dict[str, float] | None
    ) -> dict[str, float]:
        """Decide the task's criteria after `test.sh` ran (`harbor.criteria`) and fold
        them into the reward: `reward` is 1.0 when the folded score reaches the file's
        `pass_threshold`, else 0.0, and `criteria_score` is the score itself (set
        `task.reward_key = "criteria_score"` to train on it). A reward file is optional
        here; its own `reward`, when it writes one, stays as `verifier_reward`."""
        assert self.data.criteria is not None
        if scores is None:
            try:
                value = await runtime.read(
                    "/logs/verifier/reward.txt", max_bytes=MAX_REWARD_BYTES
                )
                scores = {
                    "reward": REWARD_SCALAR_ADAPTER.validate_python(
                        float(value.decode().strip())
                    )
                }
            except (SandboxError, OSError, ValueError):
                scores = {}
        report = await grade_criteria(
            self.data.criteria,
            runtime,
            trace,
            question=self.data.prompt_text,
            scores=scores,
            judge=self.config.criteria_judge,
        )
        if "reward" in scores:
            scores["verifier_reward"] = scores.pop("reward")
        return {
            **scores,
            "reward": float(report.passed),
            "criteria_score": report.score,
        }

    async def stage_trajectory(self, runtime: Runtime, trace: Trace) -> None:
        """Leave the agent's transcript and final reply where graders read them, from
        the trace alone: Harbor's `/logs/agent/trajectory.json` (CyberStrike's process
        and output-quality judges skip without it) and `/logs/agent/final_reply.md`
        (`final_reply`). `/logs/agent` is grader-owned: whatever the solver, its image
        or a restored artifact left there is removed first, so a solver cannot stand
        in for its own transcript. A trace without messages stages an empty directory;
        one without a reply stages no reply file."""
        files = {}
        if trace.messages:
            files[TRAJECTORY] = json.dumps(trajectory(trace), ensure_ascii=False)
        if reply := final_reply(trace):
            files[FINAL_REPLY] = reply
        prepare = (
            f"rm -rf {AGENT_LOGS} && mkdir -p {AGENT_LOGS} && chmod 755 {AGENT_LOGS}"
        )
        result = await runtime.run(["sh", "-c", prepare], {})
        if result.exit_code:
            raise TaskError(
                f"staging {AGENT_LOGS} failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        for path, text in files.items():
            await runtime.write(path, text.encode())
        if files:
            result = await runtime.run(["chmod", "644", *files], {})
            if result.exit_code:
                raise TaskError(f"staging {AGENT_LOGS} failed: {result.stderr.strip()}")

    async def record_verifier_output(
        self, runtime: Runtime, trace: Trace, result: ProgramResult
    ) -> None:
        """Keep what the verifier reported at `trace.info["verifier"]`: `test.sh`'s exit
        code and output, and the end of each text file it left in /logs/verifier besides
        the reward (test logs, reports). The most recently written files are kept, down
        to depth 3, so the stage that failed is among them.

        A diagnostic only: when listing the files fails, the reward reads that follow
        decide the grade, and a failed box surfaces there."""
        listing = (
            "{ for f in $(find /logs/verifier -maxdepth 3 -type f ! -name reward.json "
            "! -name reward.txt -printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 20 "
            "| cut -d' ' -f2-); do "
            'grep -Iq . "$f" 2>/dev/null || continue; '
            f'printf "==> %s <==\\n" "$f"; tail -c {VERIFIER_OUTPUT_BYTES} "$f"; echo; '
            f"done; }} | tail -c {4 * VERIFIER_OUTPUT_BYTES}"
        )
        try:
            files = (await runtime.run(["sh", "-c", listing], {})).stdout
        except SandboxError as exc:
            logger.warning("listing the verifier's logs failed: %s", exc)
            files = ""
        trace.info["verifier"] = {
            "exit_code": result.exit_code,
            "output": (result.stdout + result.stderr)[-VERIFIER_OUTPUT_BYTES:],
            "files": files,
        }
        # Scoring diagnostics outlive a failed scoring attempt (`grading`).
        trace.info.setdefault("scoring_diagnostics", {})["harbor"] = {
            "exit_code": result.exit_code,
            "stdout": result.stdout[-MAX_VERIFIER_OUTPUT_CHARS:],
            "stderr": result.stderr[-MAX_VERIFIER_OUTPUT_CHARS:],
            "stdout_truncated": len(result.stdout) > MAX_VERIFIER_OUTPUT_CHARS,
            "stderr_truncated": len(result.stderr) > MAX_VERIFIER_OUTPUT_CHARS,
        }

    async def launch_verifier(self, runtime: Runtime, trace: Trace) -> ProgramResult:
        """Run `/tests/test.sh` and return its result; `run_verifier` then reads the
        reward. By absolute path, in the runtime's configured workdir: Harbor execs the
        script the same way, and scripts do grade the agent's work at `$PWD`. A task
        that keeps the launch's output (a diagnostic report) overrides this."""
        del trace
        if self.data.verifier_user not in (None, "root", "0"):
            runtime = runtime.with_user(self.data.verifier_user)
        return await runtime.run(
            ["sh", "-c", WITHOUT_SYS_RESOURCE, "vf-verifier", "bash", "/tests/test.sh"],
            resolve_env(self.scoring.env or {}),
        )

    async def read_reward_json(self, runtime: Runtime) -> dict[str, float] | None:
        """JSON takes precedence whenever present; invalid or unreadable JSON fails.

        Only absence permits text fallback. Reads remain bounded at the source.
        """
        exists = await runtime.run(
            ["sh", "-c", 'test -e "$1" || test -L "$1"', "vf-reward", REWARD_JSON], {}
        )
        if exists.exit_code == 1:
            return None
        if exists.exit_code:
            raise TaskError(f"Cannot inspect Harbor reward file {REWARD_JSON}")
        try:
            scores = REWARD_JSON_ADAPTER.validate_json(
                await runtime.read(REWARD_JSON, max_bytes=MAX_REWARD_BYTES)
            )
        except (SandboxError, OSError, ValidationError) as exc:
            raise TaskError(
                "Harbor verifier produced invalid or unreadable reward.json; "
                "expected an object of finite numeric rewards"
            ) from exc
        return {name: float(value) for name, value in scores.items()}


WITHOUT_SYS_RESOURCE = (
    "setpriv --bounding-set -sys_resource true 2>/dev/null "
    '&& exec setpriv --bounding-set -sys_resource -- "$@"; exec "$@"'
)
"""Run the arguments without CAP_SYS_RESOURCE, as a Docker container's root runs (the
capability is not in Docker's default set). A grader that caps the code it runs with a
hard rlimit, then drops to another user through `runuser`/`su` (whose pam_limits resets
every limit to init's), keeps the cap in Docker; root on a Prime VM holds the capability,
so the cap was lifted and the grader failed its own sandbox check. Other capabilities
stay: graders use them (iptables needs CAP_NET_ADMIN). Where the bounding set cannot be
changed (no setpriv, or not root) the arguments run as they are."""


ONEDNN_GUARD = "/tmp/vf-onednn-guard.py"
PYTHONS = r"""
for py in $(command -v python3 python 2>/dev/null) /usr/local/bin/python3* /usr/bin/python3* \
    /opt/*/bin/python3* /opt/conda/bin/python3* /*/.venv/bin/python3 /app/*/.venv/bin/python3; do
  case "$py" in *python3|*python3.[0-9]|*python3.[0-9][0-9]|*/python) ;; *) continue ;; esac
  [ -x "$py" ] || continue
  real=$(readlink -f "$py" 2>/dev/null || echo "$py")
  case " $seen " in *" $real "*) continue ;; esac
  seen="$seen $real"
  printf '%s: %s\n' "$py" "$("$py" -I "$1" 2>&1 | tail -n 1)"
done
"""


async def guard_onednn(runtime: Runtime) -> str:
    """Install `onednn_guard` into each of the box's Python interpreters that has torch,
    on a CPU whose reported topology makes oneDNN divide by zero; returns what it
    installed (empty when nothing). Best effort: a box this cannot reach keeps oneDNN."""
    try:
        await runtime.write(
            ONEDNN_GUARD, (Path(__file__).parent / "onednn_guard.py").read_bytes()
        )
        result = await runtime.run(["sh", "-c", PYTHONS, "vf", ONEDNN_GUARD], {})
    except SandboxError as exc:
        logger.warning("oneDNN guard not installed: %s", exc)
        return ""
    return "\n".join(
        line for line in result.stdout.splitlines() if "installed in" in line
    )


@lru_cache(maxsize=4096)
def declared_verifier_timeout(task_dir: str | None) -> float | None:
    """The `[verifier].timeout_sec` a task's `task.toml` declares, if any."""
    if not task_dir:
        return None
    try:
        with open(Path(task_dir) / "task.toml", "rb") as file:
            timeout = tomllib.load(file).get("verifier", {}).get("timeout_sec")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return None
    return float(timeout) if timeout > 0 else None


def log_tails(files: str, size: int = 800) -> str:
    """The end of each log in a `record_verifier_output` listing, so one long log (a
    download's progress) cannot crowd out the one that names the cause."""
    tails = []
    for block in re.split(r"(?m)^(?===> .* <==$)", files):
        header, _, body = block.partition("\n")
        if header.startswith("==> "):
            tails.append(f"{header}\n{body[-size:]}")
    return "".join(tails)


def verifier_box_data(data: HarborData) -> HarborData:
    """The verifier's box, declared as task data — the rollout resolves the
    grading runtime from it (image, workdir, resources, network policy), exactly
    as the solver's box resolves from the solver task's.

    Which box follows Harbor: a declared `[verifier.environment]` states its own
    image, workdir, and resources, and what it omits is the run's default; a
    fresh copy of `[environment]` keeps the task's own. The verifier's network
    policy applies either way."""
    verifier = data.verifier
    if verifier is None:
        raise TaskError(f"task {data.name!r} declares no separate verifier")
    fresh = verifier.fresh_copy
    # A size a declared environment leaves unset is unbounded under Harbor's Docker,
    # never smaller than the agent's box; a provider's default (one small CPU) would
    # starve a grader the task sized for its environment, so an unset size is the
    # task's `[environment]` one. Resolved here, for the box, so the task's own data
    # (and its key) stays as parsed.
    sized = TaskResources(
        **{
            field: getattr(verifier.resources, field)
            if getattr(verifier.resources, field) is not None
            else getattr(data.resources, field)
            for field in ("cpu", "memory", "disk")
        },
        gpu=verifier.resources.gpu,
    )
    return data.model_copy(
        update={
            "name": f"{data.name} (verifier)",
            "image": data.verifier_image
            if data.verifier_image is not None
            else data.image,
            # A grader with an image of its own does not start from the solver's world.
            "checkpoint": None if data.verifier_image is not None else data.checkpoint,
            "workdir": data.workdir if fresh else verifier.workdir,
            "resources": data.resources if fresh else sized,
            "upload_environment": data.upload_environment
            if data.verifier_image is None
            else False,
            "env": dict(verifier.env),
            "healthcheck": verifier.healthcheck,
            "skills": [],
            "mcp_servers": [],
            "tools": [],
            "network": verifier.network,
        }
    )


def harbor_network(network) -> NetworkPolicyConfig:
    """A Harbor network mode as a network policy: `public` is unrestricted, and the
    other modes allow only their hosts (none for `no-network`)."""
    from harbor.models.task.config import NetworkMode

    if network.network_mode == NetworkMode.PUBLIC:
        return NetworkPolicyConfig()
    return NetworkPolicyConfig(allow=list(network.allowed_hosts))


def shared_grading_network(parsed) -> NetworkPolicyConfig | None:
    """The network a shared-box verifier grades under: Harbor's verifier phase policy,
    or the `[environment]` baseline when the phase declares none. None opens the box for
    grading, as for a public verifier. A restricted verifier beside a public agent phase
    also grades open: its box started without a policy, so it cannot take one."""
    from harbor.models.task.config import NetworkMode

    baseline = parsed.environment.resolve_baseline()
    grading = parsed.verifier.explicit_phase_policy() or baseline
    if grading.network_mode == NetworkMode.PUBLIC:
        return None
    agent = parsed.agent.explicit_phase_policy() or baseline
    if agent.network_mode == NetworkMode.PUBLIC:
        logger.warning(
            "Harbor task declares a %s verifier beside a public agent phase; shared "
            "grading cannot restrict a box that started open, so it grades with open "
            "network",
            grading.network_mode.value,
        )
        return None
    return harbor_network(grading)


def task_resources(environment, multiplier: float) -> TaskResources:
    """Harbor environment resource requests, scaled, as `TaskResources`.

    Harbor declares CPU counts and MB sizes; `TaskResources` wants counts and GB.
    GPU requests are never scaled.
    """
    return TaskResources(
        cpu=environment.cpus * multiplier if environment.cpus else None,
        memory=environment.memory_mb / 1024 * multiplier
        if environment.memory_mb
        else None,
        gpu=(
            f"{environment.gpu_types[0]}:{environment.gpus}"
            if environment.gpu_types
            else str(environment.gpus)
        )
        if environment.gpus
        else None,
        disk=environment.storage_mb / 1024 * multiplier
        if environment.storage_mb
        else None,
    )


def harbor_cli() -> str:
    scripts_dir = Path(sys.executable).parent
    harbor_bin = shutil.which("harbor", path=str(scripts_dir))
    if harbor_bin is None:
        raise RuntimeError(
            "Harbor tasksets require the Harbor CLI from the `harbor` extra. "
            f"Install it with: `{HARBOR_INSTALL_HINT}`"
        )
    return harbor_bin


def cache_dir(config: HarborConfig) -> Path:
    selector_parts = [config.dataset]
    if config.repo is not None:
        selector_parts.extend(("repo", config.repo))
    if config.registry_path is not None:
        registry_path = (
            config.registry_path
            if config.repo is not None
            else config.registry_path.expanduser().resolve()
        )
        selector_parts.extend(("registry_path", str(registry_path)))
    if config.registry_url is not None:
        selector_parts.extend(("registry_url", config.registry_url))
    if (
        "@" not in config.dataset
        and config.registry_path is not None
        and config.repo is None
    ):
        # A bare name selects whatever version the local registry resolves to now
        # (its `head`, say), so the cache follows what that version lists.
        selector_parts.extend(("resolved", local_registry_spec(config)))

    name = config.dataset.replace("/", "_").replace("@", "_")
    if len(selector_parts) > 1:
        digest = hashlib.sha256("\0".join(selector_parts).encode()).hexdigest()[:12]
        name = f"{name}_{digest}"
    return CACHE / name


def local_registry_spec(config: HarborConfig) -> str:
    """The dataset version a bare name resolves to in a local `registry.json`, as
    Harbor resolves it, with the tasks that version lists; empty when the file or the
    name is absent (the download then reports it)."""
    import json

    from harbor.registry.client.base import resolve_version

    try:
        registry = json.loads(config.registry_path.expanduser().read_text())
    except (OSError, ValueError):
        return ""
    specs = {
        str(spec.get("version")): spec
        for spec in registry
        if isinstance(spec, dict) and spec.get("name") == config.dataset
    }
    if not specs:
        return ""
    version = resolve_version(list(specs))
    return json.dumps(
        {"version": version, "tasks": specs[version].get("tasks")}, sort_keys=True
    )


def download_command(config: HarborConfig, output_dir: Path) -> list[str]:
    command = [
        harbor_cli(),
        "download",
        config.dataset,
        "--export",
        "-o",
        str(output_dir),
    ]
    if config.repo is not None:
        command.extend(["--repo", config.repo])
    if config.registry_path is not None:
        registry_path = (
            config.registry_path
            if config.repo is not None
            else config.registry_path.expanduser()
        )
        command.extend(["--registry-path", str(registry_path)])
    if config.registry_url is not None:
        command.extend(["--registry-url", config.registry_url])
    return command


def dataset_dir(config: HarborConfig) -> Path:
    """Download/cache a Hub or legacy-registry package selected by the config."""
    out = cache_dir(config)
    if out.is_dir():
        return out

    CACHE.mkdir(parents=True, exist_ok=True)
    # Publish only a complete CLI export to the cache.
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:
        export_dir = Path(temp) / "export"
        command = download_command(config, export_dir)
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            message = (
                f"Harbor download failed for {config.dataset!r} with exit code "
                f"{exc.returncode}"
            )
            outputs = [
                output.strip()
                for output in (exc.stdout, exc.stderr)
                if isinstance(output, str) and output.strip()
            ]
            if output := "\n".join(outputs):
                message = f"{message}:\n{output}"
            raise RuntimeError(message) from exc
        if pointers := lfs_pointers(export_dir):
            raise RuntimeError(
                f"Harbor download of {config.dataset!r} left {len(pointers)} Git LFS "
                "pointer(s) in place of their files (its `git lfs pull` failed), e.g. "
                f"{[str(p.relative_to(export_dir)) for p in pointers[:5]]}; not caching it"
            )
        try:
            export_dir.rename(out)
        except OSError:
            if out.is_dir():
                return out
            raise
    return out


LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def lfs_pointers(root: Path) -> list[Path]:
    """Files under `root` that are Git LFS pointers, not the content they stand for.

    Harbor's download ignores a failed `git lfs pull`, so an export can hold a
    pointer where a task ships a binary (programbench's gold.bin), and every grade
    from it is silently wrong."""
    found = []
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink() and path.stat().st_size < 1024:
                with path.open("rb") as f:
                    if f.read(len(LFS_POINTER)) == LFS_POINTER:
                        found.append(path)
        except OSError:
            continue
    return sorted(found)


def resolve_image(
    task_dir: Path,
    image: str | None,
    require_image: bool,
    ignore_dockerfile: bool = False,
    *,
    verifier: bool = False,
) -> str | None:
    """Choose a pullable image without silently ignoring a declared Dockerfile.

    ``None`` keeps the harness image for the solver, or the solver image for a
    separate verifier. A declared verifier environment without an image implies
    a dedicated build only when the resolved definition contains one.
    """
    if image:
        return image
    compose = task_dir / "environment" / "docker-compose.yaml"
    if not verifier and compose.is_file():
        import yaml

        main = yaml.safe_load(compose.read_text())["services"].get("main", {})
        if (
            main.get("image")
            or "build" in main
            or (task_dir / "environment/Dockerfile").is_file()
        ):
            return None
    section = "verifier.environment" if verifier else "environment"
    dockerfile = "tests/Dockerfile" if verifier else "environment/Dockerfile"
    if verifier and not (task_dir / dockerfile).is_file():
        dockerfile = "tests/docker-compose.yaml"
    if verifier or (task_dir / dockerfile).exists():
        if ignore_dockerfile:
            if verifier:
                logger.warning(
                    "%s: [%s] names no docker_image — grading in the agent's "
                    "image rather than building %s, so the verifier runs somewhere "
                    "the task never declared",
                    task_dir.name,
                    section,
                    dockerfile,
                )
            return None
        raise ValueError(
            f"{task_dir.name}: [{section}] needs a pullable docker_image instead of "
            f"building {dockerfile} — building Dockerfiles isn't supported, so this "
            "task can't run (it would otherwise score against the wrong default image). "
            "Pass --env.taskset.ignore-dockerfile to load it with a fallback image, "
            "then override the task's image or verifier_image with a pre-built image."
        )
    if require_image:
        raise ValueError(
            f"{task_dir.name}: no [{section}].docker_image and require_image=True"
        )
    return None


def parse_task(task_dir: Path, idx: int, harbor_config: HarborConfig) -> HarborData:
    # Harbor is optional, so imports stay deferred until a Harbor task loads.
    from harbor.environments.definition import should_upload_environment_dir
    from harbor.models.task.task import Task as HarborModelTask
    from harbor.models.task.verifier_mode import resolve_verifier_environment_definition

    harbor_task = HarborModelTask(task_dir)
    parsed = harbor_task.config
    if parsed.steps:
        raise ValueError(
            f"{task_dir.name}: multi-step Harbor tasks are not supported; "
            "select single-step tasks with --taskset.tasks"
        )
    artifacts, hooks, verifier = parse_verifier_extras(task_dir, parsed, harbor_config)
    environment = parsed.environment
    # Harbor ignores the `[[tools]]` table; each bundle compiles to an MCP server
    # entry, a collect hook that dumps its end state before the authored hooks (in
    # the service it runs in), and that dump as a declared artifact.
    raw = tomllib.loads((task_dir / "task.toml").read_text())
    tools = toolbundles.parse_mounts(task_dir, raw.get("tools"))
    # Harbor's own model ignores the key: `harbor run` builds or pulls as before.
    checkpoint = (raw.get("environment") or {}).get("checkpoint_id")
    if checkpoint is not None and (
        not isinstance(checkpoint, str)
        or not checkpoint
        or (task_dir / "environment" / "docker-compose.yaml").is_file()
    ):
        raise ValueError(
            f"{task_dir.name}: [environment] checkpoint_id must be a checkpoint id, "
            "for a task without environment/docker-compose.yaml"
        )
    mcp_servers = [server.model_dump(mode="json") for server in environment.mcp_servers]
    if clash := {m.bundle for m in tools} & {server["name"] for server in mcp_servers}:
        raise ValueError(
            f"{task_dir.name}: tool bundles and mcp_servers share names {sorted(clash)}"
        )
    mcp_servers += [m.mcp_server() for m in tools]
    hooks = [
        CollectHook(command=m.dump_command(), service=m.service) for m in tools
    ] + hooks
    artifacts += [HarborArtifact(source=m.dump_file, service=m.service) for m in tools]
    compose = task_dir / "environment" / "docker-compose.yaml"
    if compose.is_file() and any(m.placement == "sidecar" for m in tools):
        import yaml

        declared = set(yaml.safe_load(compose.read_text())["services"])
        if clash := sorted(
            declared & {m.bundle for m in tools if m.placement == "sidecar"}
        ):
            raise ValueError(
                f"{task_dir.name}: sidecar tool bundles {clash} share a Compose service name"
            )
    # A checkpoint carries the box: a Dockerfile beside it is how the checkpoint was
    # built, not something to run, so the task needs no image.
    image = (
        None
        if checkpoint is not None and environment.docker_image is None
        else resolve_image(
            task_dir,
            environment.docker_image,
            harbor_config.require_image,
            harbor_config.ignore_dockerfile,
        )
    )
    definition = (
        resolve_verifier_environment_definition(parsed, harbor_task.paths)
        if verifier is not None
        else None
    )
    verifier_image = (
        resolve_image(
            task_dir,
            definition.config.docker_image,
            require_image=True,
            ignore_dockerfile=harbor_config.ignore_dockerfile,
            verifier=True,
        )
        if definition is not None and definition.bundled_tests
        else None
    )
    environment_dir = task_dir / "environment"
    upload_environment = should_upload_environment_dir(
        environment_dir,
        docker_image=image,
    )
    network = parsed.agent.explicit_phase_policy() or environment.resolve_baseline()
    task, meta = parsed.task, parsed.metadata
    if "prime_kernel" in (meta or {}) and not harbor_config.boots_kernels:
        raise ValueError(
            f"{task_dir.name}: declares [metadata.prime_kernel] (it needs a kernel the "
            "provider's boxes do not run); load it with a taskset that boots it, such as "
            "sft-gen's `sft-gen-kernel-harbor`"
        )
    compose_host = ((meta or {}).get("prime_compose") or {}).get("host_image")
    if (
        compose_host is not None
        and not (environment_dir / "docker-compose.yaml").is_file()
    ):
        raise ValueError(
            f"{task_dir.name}: [metadata.prime_compose] names a host image, but the task "
            "has no environment/docker-compose.yaml"
        )
    authors = (
        [Author(name=author.name, email=author.email) for author in task.authors]
        if task
        else []
    )
    if harbor_config.ignore_timeouts:
        agent_timeout = scoring_timeout = None
    else:
        agent_timeout = (
            parsed.agent.timeout_sec
            if "timeout_sec" in parsed.agent.model_fields_set
            else None
        )
        scoring_timeout = (
            parsed.verifier.timeout_sec
            if "timeout_sec" in parsed.verifier.model_fields_set
            else None
        )
    return HarborData(
        idx=idx,
        scoring=ScoringConfig(
            mode="isolated" if verifier is not None else "shared",
            network=None
            if verifier is not None or harbor_config.shared_grading == "open"
            else shared_grading_network(parsed),
        ),
        artifact_max_bytes=harbor_config.artifact_max_bytes,
        name=harbor_task.name,
        description=task.description if task else None,
        prompt=harbor_task.instruction.strip(),
        image=image,
        workdir=environment.workdir,
        agent_user=None if parsed.agent.user is None else str(parsed.agent.user),
        network=harbor_network(network),
        timeout=TaskTimeout(
            agent=agent_timeout * harbor_config.timeout_multiplier
            if agent_timeout is not None
            else None,
            scoring=scoring_timeout * harbor_config.timeout_multiplier
            if scoring_timeout is not None
            else None,
        ),
        resources=task_resources(environment, harbor_config.resource_multiplier),
        keywords=task.keywords if task else [],
        authors=authors,
        difficulty=meta.get("difficulty"),
        category=meta.get("category"),
        tags=meta.get("tags", []),
        task_dir=str(task_dir),
        upload_environment=upload_environment,
        skills=[{"runtime": environment.skills_dir}] if environment.skills_dir else [],
        **environment.model_dump(include={"env", "healthcheck"}, mode="json"),
        mcp_servers=mcp_servers,
        tools=tools,
        checkpoint=checkpoint,
        compose_host_image=compose_host,
        verifier_image=verifier_image,
        verifier_env=parsed.verifier.env,
        artifacts=artifacts,
        collect=hooks,
        verifier=verifier,
        verifier_user=None
        if parsed.verifier.user is None
        else str(parsed.verifier.user),
        criteria=load_task_criteria(task_dir),
    )


def parse_verifier_extras(
    task_dir: Path, parsed, harbor_config: HarborConfig
) -> tuple[list[HarborArtifact], list[CollectHook], VerifierConfig | None]:
    """Harbor's `artifacts`, `[[verifier.collect]]` blocks, and verifier environment,
    narrowed to what verifiers' verifier-runtime integration can honor.

    The convention dir is deliberately not prepended here (Harbor's
    `with_convention_entry` would): collection injects it itself, as an optional sweep.
    Prepending it would make it an explicitly declared entry, and declared entries are
    required — which would fail every task that never writes there.
    """
    from harbor.models.task.artifacts import (
        effective_artifact_service,
        normalize_artifact_entries,
    )

    verifier = parsed.verifier

    artifacts: list[HarborArtifact] = []
    for entry in normalize_artifact_entries(parsed.artifacts):
        artifacts.append(
            HarborArtifact(
                source=entry.source,
                destination=entry.destination,
                service=effective_artifact_service(entry),
                exclude=list(entry.exclude or []),
                required=False,
            )
        )

    # An artifact that would restore into a grader-owned path is left out of this task
    # alone: the grader stages those paths itself (`/logs/agent` from the trace), so the
    # artifact could only overwrite them. Refusing the task here would fail the load of
    # every task in its taskset.
    if owned := grader_owned(artifact.source for artifact in artifacts):
        logger.warning(
            "%s: artifacts %s would restore into the grader's own paths %s; left out, "
            "the grader stages those itself",
            task_dir.name,
            owned,
            list(GRADER_PATHS),
        )
        artifacts = [a for a in artifacts if a.source not in owned]

    hooks: list[CollectHook] = []
    for hook in verifier.collect:
        hooks.append(
            CollectHook(
                command=hook.command,
                timeout_sec=hook.timeout_sec,
                service=hook.service,
                user=None if hook.user is None else str(hook.user),
            )
        )
    services = {entry.service for entry in (*artifacts, *hooks)} - {"main"}
    if services:
        compose = task_dir / "environment" / "docker-compose.yaml"
        declared: set[str] = set()
        if compose.is_file():
            import yaml

            declared = set(yaml.safe_load(compose.read_text())["services"])
        if missing := services - declared:
            raise ValueError(
                f"{task_dir.name}: artifacts or collect hooks target services "
                f"{sorted(missing)} that environment/docker-compose.yaml does not declare"
            )

    return artifacts, hooks, parse_verifier_environment(task_dir, parsed, harbor_config)


def parse_verifier_environment(
    task_dir: Path, parsed, harbor_config: HarborConfig
) -> VerifierConfig | None:
    """The box Harbor wants this task's verifier in, or None to grade in the agent's.

    Harbor resolves `[verifier.environment]` if declared, else a deep copy of
    `[environment]` — so a mode-only `separate` lands on the task's own image and needs
    nothing but a second box. A declared environment is the case that can name a
    different image or only override resources. Image selection is resolved separately
    by Harbor: verifier image, tests build definition, then the agent environment.
    """
    from harbor.models.task.config import TaskOS
    from harbor.models.task.paths import TaskPaths
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_task_verifier_mode,
        resolve_verifier_environment_definition,
    )

    if resolve_task_verifier_mode(parsed) != VerifierEnvironmentMode.SEPARATE:
        return None
    if harbor_config.ignore_separate_verifier:
        logger.warning(
            "%s: asks for a separate verifier; grading in the agent's box anyway "
            "(--taskset.ignore-separate-verifier)",
            task_dir.name,
        )
        return None

    definition = resolve_verifier_environment_definition(parsed, TaskPaths(task_dir))
    if definition is None:  # unreachable while the mode is SEPARATE
        raise ValueError(f"{task_dir.name}: separate verifier resolved no environment")
    environment = definition.config
    declared = parsed.verifier.environment is not None
    unsupported = [field for field in ("tpu",) if getattr(environment, field, None)]
    if environment.os != TaskOS.LINUX or unsupported:
        raise ValueError(
            f"{task_dir.name}: verifier environment declares "
            f"{unsupported or environment.os}, which verifiers' verifier-runtime "
            "integration cannot honor"
        )

    network = parsed.verifier.explicit_phase_policy() or environment.resolve_baseline()
    return VerifierConfig(
        # A declared environment states its own resources; what it leaves out is filled
        # when the verifier's box is built (`verifier_box_data`). A fresh copy is the
        # task's environment, so it keeps whatever the agent box resolved to.
        resources=(
            task_resources(environment, harbor_config.resource_multiplier)
            if declared
            else TaskResources()
        ),
        workdir=environment.workdir if declared else None,
        **environment.model_dump(include={"env", "healthcheck"}, mode="json"),
        fresh_copy=not declared,
        network=harbor_network(network),
    )


# Downloaded task directories are immutable. Cache the current task's environment
# and tests to bound memory while reusing both archives across rollouts.
@lru_cache(maxsize=2)
def make_tar(directory: Path) -> bytes:
    """`directory`'s contents, root-owned as Harbor's copy into a container leaves them,
    with the modes git records (no group or other write, whatever this checkout's
    umask): a grader that trusts only root-owned, unwritable files under /tests
    (abundant-programbench wim-archive) refuses ones carrying the host's uid or a
    group-writable mode."""

    def as_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        info.mode &= ~0o022
        return info

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name, filter=as_root)
    return buffer.getvalue()


class HarborTaskset(Taskset[HarborTask, HarborConfig]):
    def load(self) -> Iterator[HarborTask]:
        root = dataset_dir(self.config)
        found = sorted(toml_path.parent for toml_path in root.rglob("task.toml"))
        # A task.toml inside another task's folder (a fixture its environment ships, such
        # as a nested task definition its tests use) is part of that task, not a task.
        tasks = set(found)
        task_dirs = [
            task_dir
            for task_dir in found
            if not any(parent in tasks for parent in task_dir.parents)
            and (self.config.tasks is None or task_dir.name in self.config.tasks)
        ]
        if not task_dirs:
            raise ValueError(f"no harbor tasks found in {root}")
        for idx, task_dir in enumerate(task_dirs):
            yield HarborTask(parse_task(task_dir, idx, self.config), self.config.task)


def grader_owned(roots) -> list[str]:
    """The artifact roots that would restore into a grader-owned path."""
    return sorted(
        root
        for root in roots
        if any(PurePosixPath(root).is_relative_to(path) for path in GRADER_PATHS)
    )


def text(content: Any) -> str:
    """A message's content as text; a non-text part as its type."""
    if content is None or isinstance(content, str):
        return content or ""
    parts = []
    for part in content:
        part = part if isinstance(part, dict) else part.model_dump()
        parts.append(part.get("text") or f"[{part.get('type', 'content')}]")
    return "\n".join(parts)


def final_reply(trace: Trace) -> str:
    """The solver's final reply: the agent's last response (`Trace.last_reply`), else
    the last assistant message on the trace's final branch (a gold solution's reply,
    which no model sampled; see `validation.GOLD_REPLY`)."""
    if reply := trace.last_reply:
        return reply
    for message in reversed(trace.messages):
        if message.role == "assistant":
            return text(message.content).strip()
    return ""


def trajectory(trace: Trace) -> dict[str, Any]:
    """The trace's final branch as a Harbor (ATIF-style) trajectory: one step per
    message, its text, and an agent step's tool calls."""
    sources = {"assistant": "agent"}
    steps = []
    for index, message in enumerate(trace.messages, start=1):
        step: dict[str, Any] = {
            "step_id": index,
            "source": sources.get(message.role, message.role),
            "message": text(message.content),
        }
        for call in getattr(message, "tool_calls", None) or []:
            step.setdefault("tool_calls", []).append(
                {
                    "tool_call_id": call.id,
                    "function_name": call.name,
                    "arguments": call.arguments,
                }
            )
        if message.role == "tool":
            step["tool_call_id"] = message.tool_call_id
        steps.append(step)
    return {"schema_version": "ATIF-v1.2", "session_id": trace.id, "steps": steps}
