"""Harbor tasksets backed by Harbor Hub packages.

The Harbor CLI downloads and caches each task directory. Its verifier runs in the
runtime the harness edited — or, when the task asks for it with
``[verifier].environment_mode = "separate"``, in a second box the agent never
touched, carrying only what the task declared — the harbor env provisions and
grades that box (see ``env.py``). Either way the score lands in
``/logs/verifier/reward.json`` or the legacy ``reward.txt``.

A pullable ``[environment].docker_image`` becomes ``TaskData.image``. Verifiers does
not build Dockerfile-only environments, so those are rejected unless ``ignore_dockerfile``
deliberately uses the harness runtime image. Tasks without an environment also use that
image unless ``require_image`` is set. The same rule applies to a declared
``[verifier.environment]``: it needs a pullable ``docker_image``, since Harbor would
otherwise build the verifier image from ``tests/Dockerfile``.
"""

import asyncio
import copy
import hashlib
import io
import logging
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from verifiers.v1.configs.taskset import TasksetConfig
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskData, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset
from verifiers.v1.trace import Trace
from verifiers.v1.utils.artifacts import Artifact, collect
from verifiers.v1.utils.decorators import reward

logger = logging.getLogger(__name__)

CACHE = Path.home() / ".cache" / "harbor"
HARBOR_INSTALL_HINT = "uv sync --python 3.12 --extra harbor"
REWARD_JSON = "/logs/verifier/reward.json"
MAX_REWARD_BYTES = 1024 * 1024
REWARD_JSON_ADAPTER = TypeAdapter(
    float | Annotated[dict[str, float], Field(min_length=1)],
    config=ConfigDict(strict=True, allow_inf_nan=False),
)


class HarborConfig(TasksetConfig):
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
    """Drop each task's declared agent and verifier timeouts so rollouts run
    unbounded (unless run-level `--timeout.*` limits are set). Task timeouts are
    authored against Harbor's runtime and confound model capability with inference
    speed; set False to apply them anyway."""
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
    ignore_separate_verifier: bool = False
    """Grade every task in the agent's own box, even one whose `[verifier]` asks for a
    separate one. Trades the isolation for a sandbox per task; useful when provisioning
    is the bottleneck. Note what it gives up: the grader becomes reachable by the agent
    that just ran."""


class Author(BaseModel):
    name: str | None = None
    email: str | None = None


class CollectHook(BaseModel):
    """One `[[verifier.collect]]` command, run in the agent's box by `finalize`."""

    command: str
    timeout_sec: float = 600.0


class VerifierConfig(BaseModel):
    """The box this task's verifier wants, when it wants one of its own.

    `None` on `HarborData` means shared — grade where the agent worked, which is still
    Harbor's default and every task that says nothing."""

    image: str | None = None
    """Pullable ref from `[verifier.environment].docker_image`. None keeps the task's
    own image, which is what Harbor's fresh copy of `[environment]` resolves to."""
    resources: TaskResources = TaskResources()
    workdir: str | None = None
    fresh_copy: bool = False
    """Whether this came from Harbor's fresh copy of `[environment]` rather than a
    declared `[verifier.environment]`. A fresh copy inherits the agent box's resolved
    resources; a declared environment states its own, and what it omits falls back to
    the run's rather than to the agent's task-derived values."""
    network_allow: list[str] = Field(default_factory=lambda: ["*"])
    """Destinations the verifier may reach, from the verifier's network mode. `["*"]`
    is unrestricted; `[]` is Harbor's `no-network` / `allow_internet = false`."""


class StepHealthcheck(BaseModel):
    """Harbor's Docker-style healthcheck contract, serialized with the task row."""

    command: str
    interval_sec: float = 5.0
    timeout_sec: float = 30.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 5.0
    retries: int = 3


class HarborStep(BaseModel):
    """One parsed ``[[steps]]`` entry and its effective verifier contract."""

    name: str
    prompt: str
    timeout: TaskTimeout = TaskTimeout()
    verifier_env: dict[str, str] = Field(default_factory=dict)
    collect: list[CollectHook] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    verifier: VerifierConfig | None = None
    min_reward: float | dict[str, float] | None = None
    healthcheck: StepHealthcheck | None = None


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
    verifier_env: dict[str, str] = Field(default_factory=dict)
    """Raw [verifier.env] entries (literals or `${VAR}`/`${VAR:-default}` templates).
    Resolved against the host environment at scoring time, like `harbor run` — so a
    verifier that needs judge API keys or configuration actually receives them."""
    collect: list[CollectHook] = Field(default_factory=list)
    """`[[verifier.collect]]` blocks: commands that snapshot runtime state into files
    after the agent stops, so the files can travel to a grading box as artifacts."""
    verifier: VerifierConfig | None = None
    """The verifier's own box, when `[verifier].environment_mode` asks for one. None
    grades in the agent's box."""
    steps: list[HarborStep] = Field(default_factory=list)
    """Ordered Harbor steps. Empty keeps the ordinary single-step path."""
    current_step: str | None = None
    """The step represented by an env-minted rollout task."""
    step_healthcheck: StepHealthcheck | None = None
    environment_healthcheck: StepHealthcheck | None = None
    multi_step_reward_strategy: Literal["mean", "final"] = "mean"
    resume_session: bool = False
    """Whether one interaction is carrying the agent across all steps."""


class HarborTask(Task[HarborData]):
    """Stage and run Harbor's verifier inside the task's live runtime."""

    NEEDS_CONTAINER = True

    def for_step(
        self,
        step: HarborStep,
        *,
        first: bool = False,
        resume_session: bool = False,
    ) -> "HarborTask":
        """Mint one rollout task while retaining the trial's shared runtime shape."""
        data = self.data.model_copy(
            update={
                "prompt": step.prompt,
                "timeout": step.timeout,
                "verifier_env": step.verifier_env,
                "collect": step.collect,
                "artifacts": step.artifacts,
                "verifier": step.verifier,
                "current_step": step.name,
                "step_healthcheck": step.healthcheck,
                "upload_environment": self.data.upload_environment if first else False,
                "resume_session": resume_session,
                "steps": self.data.steps if resume_session else [],
            }
        )
        return type(self)(data, self.config)

    async def setup(self, runtime: Runtime) -> None:
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
        if self.data.current_step is not None:
            await stage_step_workdir(
                runtime, self.data.task_dir, self.data.current_step
            )
        if self.data.step_healthcheck is not None:
            await run_healthcheck(runtime, self.data.step_healthcheck)

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        """Run Harbor's collect hooks while the agent's box is still alive.

        Harbor runs these after the agent phase and before artifact collection, which
        is exactly what `finalize` means here, so the hook maps onto the existing
        lifecycle rather than needing a stage of its own.

        Strict, unlike `harbor run`, which logs a failed hook and carries on: there the
        output is observability, here it is a grading input, and a silently absent file
        makes the verifier score a stale state instead of failing loudly.
        """
        if self.data.resume_session:
            # The env collects and grades between interaction turns.
            return
        await self.collect_step(trace, runtime)

    async def collect_step(self, trace: Trace, runtime: Runtime) -> None:
        """Run effective collect hooks and snapshot this step's artifacts."""
        for hook in self.data.collect:
            try:
                result = await asyncio.wait_for(
                    runtime.run(["sh", "-c", hook.command], {}),
                    hook.timeout_sec,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    f"collect hook timed out after {hook.timeout_sec}s: {hook.command}"
                ) from exc
            if result.exit_code:
                detail = (result.stderr or result.stdout).strip()[-500:]
                raise RuntimeError(
                    f"collect hook failed (exit {result.exit_code}): "
                    f"{hook.command}\n{detail}"
                )
        trace.state.artifacts = await collect(runtime, self.data.artifacts)

    def graded_elsewhere(self) -> "HarborTask":
        """A copy whose `solved` records nothing here: the harbor env grades this
        task's finished work in a separate box of the task's choosing."""
        clone = copy.copy(self)
        clone._graded_elsewhere = True
        return clone

    _graded_elsewhere: bool = False

    async def _stage_tests(self, runtime: Runtime, wipe: bool = False) -> None:
        """Put shared tests then the current step's overlay into ``/tests``.

        Raises rather than scoring stale state: a leftover reward file — planted by
        the agent or shipped in the image — must be gone before `test.sh` runs, so a
        removal that fails must not fall through to reading it.

        `wipe` for a box we did not watch being built: a fresh container of the task's
        image can ship its own `/tests`, and a leftover file there would be graded as
        though it came from the package.
        """
        task_dir = Path(self.data.task_dir)
        sources = [task_dir / "tests"]
        if self.data.current_step is not None:
            sources.append(task_dir / "steps" / self.data.current_step / "tests")
        sources = [source for source in sources if source.is_dir()]
        stage = (
            f"{'rm -rf /tests && ' if wipe or self.data.current_step else ''}"
            "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt && "
            "mkdir -p /logs/verifier /tests"
        )
        result = await runtime.run(["sh", "-c", stage], {})
        if result.exit_code:
            raise TaskError(
                f"staging tests failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        for index, source in enumerate(sources):
            archive = f"/tmp/harbor-tests-{index}.tgz"
            await runtime.write(archive, make_tar(source))
            result = await runtime.run(
                [
                    "sh",
                    "-c",
                    f"tar --no-same-owner -xzf {shlex.quote(archive)} -C /tests",
                ],
                {},
            )
            if result.exit_code:
                raise TaskError(
                    f"staging tests from {source} failed (exit {result.exit_code}): "
                    f"{(result.stderr or result.stdout).strip()[-500:]}"
                )

    @reward(weight=1.0)
    async def solved(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        if self.data.resume_session:
            return aggregate_step_rewards(
                trace.info.get("harbor_steps", []),
                self.data.multi_step_reward_strategy,
            )
        if self.data.verifier is not None:
            if not self._graded_elsewhere:
                raise TaskError(
                    f"task {self.data.name!r} declares a separate verifier "
                    '([verifier].environment_mode = "separate"); grade it through '
                    "the harbor env (this taskset's default), or force shared "
                    "grading with --taskset.ignore-separate-verifier"
                )
            return {}
        await self._stage_tests(runtime)
        if self.data.current_step is not None:
            scores = await self._step_graded(runtime)
            trace.info["harbor_step"] = self.data.current_step
            trace.info["harbor_step_rewards"] = scores
            return scores
        return await self._graded(runtime, trace)

    async def _step_graded(self, runtime: Runtime) -> dict[str, float]:
        """Run one step verifier while preserving every Harbor reward dimension."""
        await runtime.run(["bash", "/tests/test.sh"], verifier_env(self.data))
        scores = await self._reward_json(runtime)
        if scores is not None:
            if isinstance(scores, dict):
                return scores
            return {"reward": scores}
        try:
            reward = (
                (
                    await runtime.read(
                        "/logs/verifier/reward.txt", max_bytes=MAX_REWARD_BYTES
                    )
                )
                .decode()
                .strip()
            )
            if not reward:
                raise ValueError("reward.txt is empty")
            return {"reward": float(reward)}
        except (SandboxError, OSError, ValueError) as exc:
            raise TaskError(
                "Harbor step verifier produced no valid reward.json or reward.txt"
            ) from exc

    async def _graded(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        # By absolute path, in the runtime's configured workdir: Harbor execs the
        # script the same way, and scripts do grade the agent's work at `$PWD`.
        await runtime.run(["bash", "/tests/test.sh"], verifier_env(self.data))
        scores = await self._reward_json(runtime)
        if scores is not None:
            if isinstance(scores, dict) and "reward" in scores:
                trace.record_metrics(
                    {key: value for key, value in scores.items() if key != "reward"}
                )
                return {"reward": scores["reward"]}
            return scores
        try:
            reward = (
                (
                    await runtime.read(
                        "/logs/verifier/reward.txt", max_bytes=MAX_REWARD_BYTES
                    )
                )
                .decode()
                .strip()
            )
            return float(reward or 0)
        except (SandboxError, OSError, ValueError):
            return 0.0

    async def _reward_json(self, runtime: Runtime) -> float | dict[str, float] | None:
        """Read Harbor's scalar or keyed JSON reward, if it is valid.

        Bounded: this is a grading input, and nothing guarantees its size.
        """
        try:
            return REWARD_JSON_ADAPTER.validate_json(
                await runtime.read(REWARD_JSON, max_bytes=MAX_REWARD_BYTES)
            )
        except (SandboxError, OSError, ValidationError):
            return None


def verifier_box_data(data: HarborData) -> HarborData:
    """The verifier's box, declared as task data — the harbor env resolves the
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
    return data.model_copy(
        update={
            "name": f"{data.name} (verifier)",
            "image": verifier.image if verifier.image is not None else data.image,
            "workdir": data.workdir if fresh else verifier.workdir,
            "resources": data.resources if fresh else verifier.resources,
            "network_allow": list(verifier.network_allow),
            "network_block": [],
        }
    )


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
        gpu=str(environment.gpus) if environment.gpus else None,
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

    name = config.dataset.replace("/", "_").replace("@", "_")
    if len(selector_parts) > 1:
        digest = hashlib.sha256("\0".join(selector_parts).encode()).hexdigest()[:12]
        name = f"{name}_{digest}"
    return CACHE / name


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
        try:
            export_dir.rename(out)
        except OSError:
            if out.is_dir():
                return out
            raise
    return out


def resolve_image(
    task_dir: Path,
    image: str | None,
    require_image: bool,
    ignore_dockerfile: bool = False,
) -> str | None:
    """Choose a pullable image without silently ignoring a declared Dockerfile.

    ``None`` tells the runtime to keep the harness image. That is the intended
    fallback for tasks with no environment, but would score a Dockerfile task in
    the wrong environment unless the user explicitly opts in.
    """
    if image:
        return image
    if (task_dir / "environment" / "Dockerfile").exists():
        if ignore_dockerfile:
            return None
        raise ValueError(
            f"{task_dir.name}: environment is a Dockerfile, not a pullable "
            "[environment].docker_image — building Dockerfiles isn't supported, so this "
            "task can't run (it would otherwise score against the wrong default image). "
            "Pass --env.taskset.ignore-dockerfile to run it on the harness runtime's image instead."
        )
    if require_image:
        raise ValueError(
            f"{task_dir.name}: no [environment].docker_image and require_image=True"
        )
    return None


def parse_task(task_dir: Path, idx: int, harbor_config: HarborConfig) -> HarborData:
    # Harbor is optional, so imports stay deferred until a Harbor task loads.
    from harbor.environments.definition import should_upload_environment_dir
    from harbor.models.task.config import NetworkMode
    from harbor.models.task.task import Task as HarborModelTask

    harbor_task = HarborModelTask(task_dir)
    parsed = harbor_task.config
    artifacts, hooks, verifier = parse_verifier_extras(task_dir, parsed, harbor_config)
    steps = parse_steps(task_dir, harbor_task, parsed, harbor_config)
    environment = parsed.environment
    environment_dir = task_dir / "environment"
    upload_environment = should_upload_environment_dir(
        environment_dir,
        docker_image=environment.docker_image,
    )
    network = parsed.agent.explicit_phase_policy() or environment.resolve_baseline()
    task, meta = parsed.task, parsed.metadata
    authors = (
        [Author(name=author.name, email=author.email) for author in task.authors]
        if task
        else []
    )
    # Older registry entries stored one author in [metadata].
    if not authors and meta.get("author_name"):
        authors = [Author(name=meta["author_name"], email=meta.get("author_email"))]
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
        name=harbor_task.name,
        description=task.description if task else None,
        prompt=None if steps else harbor_task.instruction.strip(),
        image=resolve_image(
            task_dir,
            environment.docker_image,
            harbor_config.require_image,
            harbor_config.ignore_dockerfile,
        ),
        workdir=environment.workdir,
        network_allow=(
            ["*"]
            if network.network_mode == NetworkMode.PUBLIC
            else list(network.allowed_hosts)
        ),
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
        verifier_env=parsed.verifier.env,
        artifacts=artifacts,
        collect=hooks,
        verifier=None if steps else verifier,
        steps=steps,
        environment_healthcheck=(
            StepHealthcheck.model_validate(environment.healthcheck.model_dump())
            if environment.healthcheck is not None
            else None
        ),
        multi_step_reward_strategy=(
            parsed.multi_step_reward_strategy.value
            if parsed.multi_step_reward_strategy is not None
            else "mean"
        ),
    )


def parse_steps(
    task_dir: Path, harbor_task, parsed, config: HarborConfig
) -> list[HarborStep]:
    """Translate Harbor steps and resolve each step's effective verifier shape."""
    steps: list[HarborStep] = []
    for step in parsed.steps or []:
        if step.agent.user is not None or parsed.agent.user is not None:
            raise ValueError(
                f"{task_dir.name}: step {step.name!r} declares an agent user; "
                "verifiers runtimes do not support per-command users"
            )
        if step.agent.explicit_phase_policy() is not None:
            raise ValueError(
                f"{task_dir.name}: step {step.name!r} declares an agent network "
                "policy override, which cannot be applied to the shared runtime"
            )
        verifier_user = (
            step.verifier.user
            if step.verifier.user is not None
            else parsed.verifier.user
        )
        if verifier_user is not None:
            raise ValueError(
                f"{task_dir.name}: step {step.name!r} declares a verifier user; "
                "verifiers runtimes do not support per-command users"
            )
        verifier = parse_verifier_environment(task_dir, parsed, config, step=step)
        verifier_policy = (
            step.verifier.explicit_phase_policy()
            or parsed.verifier.explicit_phase_policy()
        )
        if verifier is None and verifier_policy is not None:
            raise ValueError(
                f"{task_dir.name}: step {step.name!r} declares a shared-verifier "
                "network override, which cannot replace the live agent policy"
            )

        if config.ignore_timeouts:
            agent_timeout = verifier_timeout = None
        else:
            agent_timeout = (
                step.agent.timeout_sec
                if step.agent.timeout_sec is not None
                else parsed.agent.timeout_sec
            )
            verifier_timeout = (
                step.verifier.timeout_sec
                if step.verifier.timeout_sec is not None
                else parsed.verifier.timeout_sec
            )
        steps.append(
            HarborStep(
                name=step.name,
                prompt=harbor_task.step_instruction(step.name).strip(),
                timeout=TaskTimeout(
                    agent=(
                        agent_timeout * config.timeout_multiplier
                        if agent_timeout is not None
                        else None
                    ),
                    scoring=(
                        verifier_timeout * config.timeout_multiplier
                        if verifier_timeout is not None
                        else None
                    ),
                ),
                verifier_env={**parsed.verifier.env, **step.verifier.env},
                collect=parse_collect_hooks(
                    task_dir, [*parsed.verifier.collect, *step.verifier.collect]
                ),
                artifacts=parse_artifacts(
                    task_dir, [*parsed.artifacts, *step.artifacts]
                ),
                verifier=verifier,
                min_reward=step.min_reward,
                healthcheck=(
                    StepHealthcheck.model_validate(step.healthcheck.model_dump())
                    if step.healthcheck is not None
                    else None
                ),
            )
        )
    return steps


def parse_verifier_extras(
    task_dir: Path, parsed, harbor_config: HarborConfig
) -> tuple[list[Artifact], list[CollectHook], VerifierConfig | None]:
    """Harbor's `artifacts`, `[[verifier.collect]]` blocks, and verifier environment,
    narrowed to what verifiers' verifier-runtime integration can honor.

    The convention dir is deliberately not prepended here (Harbor's
    `with_convention_entry` would): collection injects it itself, as an optional sweep.
    Prepending it would make it an explicitly declared entry, and declared entries are
    required — which would fail every task that never writes there.
    """
    verifier = parsed.verifier
    if verifier.user is not None:
        raise ValueError(f"{task_dir.name}: [verifier].user is not supported")

    return (
        parse_artifacts(task_dir, parsed.artifacts),
        parse_collect_hooks(task_dir, verifier.collect),
        parse_verifier_environment(task_dir, parsed, harbor_config),
    )


def parse_artifacts(task_dir: Path, entries) -> list[Artifact]:
    from harbor.constants import MAIN_SERVICE_NAME
    from harbor.models.task.artifacts import (
        effective_artifact_service,
        normalize_artifact_entries,
    )

    artifacts: list[Artifact] = []
    for entry in normalize_artifact_entries(entries):
        if effective_artifact_service(entry) != MAIN_SERVICE_NAME:
            raise ValueError(
                f"{task_dir.name}: artifact {entry.source!r} targets additional "
                f"service {entry.service!r}; verifiers currently supports artifacts "
                "from the main service only"
            )
        # `destination` positions a file in Harbor's host trial directory. Verifiers has
        # no such directory (the trace is the record) and Harbor never lets destination
        # affect verifier-side placement, so it cannot change any grading outcome.
        artifacts.append(
            Artifact(
                source=entry.source,
                exclude=list(entry.exclude or []),
                required=False,
            )
        )

    return artifacts


def parse_collect_hooks(task_dir: Path, entries) -> list[CollectHook]:
    from harbor.constants import MAIN_SERVICE_NAME

    hooks: list[CollectHook] = []
    for hook in entries:
        if hook.service != MAIN_SERVICE_NAME:
            raise ValueError(
                f"{task_dir.name}: collect hook targets additional service "
                f"{hook.service!r}; verifiers currently supports collect hooks for "
                "the main service only"
            )
        if hook.user is not None:
            raise ValueError(
                f"{task_dir.name}: collect hook `user` is not supported "
                "(commands run as the runtime's default user)"
            )
        hooks.append(CollectHook(command=hook.command, timeout_sec=hook.timeout_sec))

    return hooks


def parse_verifier_environment(
    task_dir: Path, parsed, harbor_config: HarborConfig, *, step=None
) -> VerifierConfig | None:
    """The box Harbor wants this task's verifier in, or None to grade in the agent's.

    Harbor resolves `[verifier.environment]` if declared, else a deep copy of
    `[environment]` — so a mode-only `separate` lands on the task's own image and needs
    nothing but a second box. A declared environment is the case that can name a
    different image, and the case that can name none at all: there Harbor builds
    `tests/Dockerfile`, which verifiers never does.
    """
    from harbor.models.task.config import NetworkMode, TaskOS
    from harbor.models.task.verifier_mode import (
        VerifierEnvironmentMode,
        resolve_effective_verifier_env_config,
        resolve_step_verifier_mode,
        resolve_task_verifier_mode,
    )

    mode = (
        resolve_step_verifier_mode(parsed, step)
        if step is not None
        else resolve_task_verifier_mode(parsed)
    )
    if mode != VerifierEnvironmentMode.SEPARATE:
        return None
    if harbor_config.ignore_separate_verifier:
        logger.warning(
            "%s%s: asks for a separate verifier; grading in the agent's box anyway "
            "(--taskset.ignore-separate-verifier)",
            task_dir.name,
            f" step {step.name!r}" if step is not None else "",
        )
        return None

    environment = resolve_effective_verifier_env_config(parsed, step)
    if environment is None:  # unreachable while the mode is SEPARATE
        raise ValueError(f"{task_dir.name}: separate verifier resolved no environment")
    declared = parsed.verifier.environment is not None or (
        step is not None and step.verifier.environment is not None
    )

    if declared and environment.docker_image is None:
        if not harbor_config.ignore_dockerfile:
            raise ValueError(
                f"{task_dir.name}: [verifier.environment] names no docker_image, so "
                "Harbor would build the verifier image from tests/Dockerfile. Verifiers "
                "pulls images and never builds them: build and push it yourself (e.g. "
                "`prime images push`) and set [verifier.environment].docker_image to the "
                "resulting ref, or pass --taskset.ignore-dockerfile to grade in the "
                "agent's image instead."
            )
        logger.warning(
            "%s: [verifier.environment] names no docker_image — grading in the agent's "
            "image rather than building tests/Dockerfile, so the verifier runs somewhere "
            "the task never declared",
            task_dir.name,
        )
    unsupported = [
        field
        for field in ("healthcheck", "mcp_servers", "skills_dir", "gpu_types", "tpu")
        if getattr(environment, field, None)
    ]
    if environment.os != TaskOS.LINUX or unsupported:
        raise ValueError(
            f"{task_dir.name}: verifier environment declares "
            f"{unsupported or environment.os}, which verifiers' verifier-runtime "
            "integration cannot honor"
        )

    network = (
        (step.verifier.explicit_phase_policy() if step is not None else None)
        or parsed.verifier.explicit_phase_policy()
        or environment.resolve_baseline()
    )
    return VerifierConfig(
        image=environment.docker_image if declared else None,
        # A declared environment states its own resources; what it leaves out is the
        # run's default, not the agent task's. A fresh copy is the task's environment,
        # so it keeps whatever the agent box resolved to.
        resources=(
            task_resources(environment, harbor_config.resource_multiplier)
            if declared
            else TaskResources()
        ),
        workdir=environment.workdir if declared else None,
        fresh_copy=not declared,
        network_allow=(
            ["*"]
            if network.network_mode == NetworkMode.PUBLIC
            else list(network.allowed_hosts)
        ),
    )


def verifier_env(task: HarborData) -> dict[str, str]:
    """Resolve templates at scoring time so host secrets are never serialized."""
    if not task.verifier_env:
        return {}

    # Harbor is an optional dependency, so importing this module must still work
    # for users who do not install the Harbor extra.
    from harbor.utils.env import resolve_env_vars

    return resolve_env_vars(task.verifier_env)


# Downloaded task directories are immutable. Keep a small working set because a
# multi-step task may alternate shared tests and several step overlays.
@lru_cache(maxsize=8)
def make_tar(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name)
    return buffer.getvalue()


async def stage_step_workdir(runtime: Runtime, task_dir: str, step_name: str) -> None:
    """Overlay ``steps/{name}/workdir`` and run its reserved setup script."""
    directory = Path(task_dir) / "steps" / step_name / "workdir"
    if not directory.is_dir():
        return
    archive = (
        f"/tmp/harbor-step-{hashlib.sha256(step_name.encode()).hexdigest()[:12]}.tgz"
    )
    await runtime.write(archive, make_tar(directory))
    workdir = getattr(runtime.config, "workdir", None) or "/"
    result = await runtime.run(
        [
            "sh",
            "-c",
            f"tar --no-same-owner -xzf {shlex.quote(archive)} -C {shlex.quote(workdir)}",
        ],
        {},
    )
    if result.exit_code:
        raise RuntimeError(
            f"failed to stage Harbor step {step_name!r}: "
            f"{(result.stderr or result.stdout).strip()[-500:]}"
        )
    if (directory / "setup.sh").is_file():
        result = await runtime.run(["bash", f"{workdir.rstrip('/')}/setup.sh"], {})
        if result.exit_code:
            raise RuntimeError(
                f"step {step_name!r} setup.sh exited with {result.exit_code}: "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )


async def run_healthcheck(runtime: Runtime, config: StepHealthcheck) -> None:
    """Mirror Harbor's consecutive-retry and start-period healthcheck semantics."""
    start_period_end = time.monotonic() + config.start_period_sec
    failures = 0
    while True:
        try:
            result = await asyncio.wait_for(
                runtime.run(["sh", "-c", config.command], {}), config.timeout_sec
            )
        except TimeoutError:
            result = None
        if result is not None and result.exit_code == 0:
            return
        if time.monotonic() < start_period_end:
            await asyncio.sleep(config.start_interval_sec)
            continue
        failures += 1
        if failures >= config.retries:
            raise RuntimeError(
                f"healthcheck failed after {config.retries} consecutive retries: "
                f"{config.command}"
            )
        await asyncio.sleep(config.interval_sec)


def aggregate_step_rewards(
    results: list[dict], strategy: Literal["mean", "final"]
) -> dict[str, float]:
    """Apply Harbor's trial-level reward strategy to completed verifier results."""
    if strategy == "final":
        if not results or results[-1].get("rewards") is None:
            return {}
        return {key: float(value) for key, value in results[-1]["rewards"].items()}
    valid = [
        result["rewards"] for result in results if result.get("rewards") is not None
    ]
    if not valid:
        return {}
    keys = {key for rewards in valid for key in rewards}
    return {
        key: sum(float(rewards.get(key, 0)) for rewards in valid) / len(valid)
        for key in keys
    }


def min_reward_failure(
    rewards: dict[str, float] | None,
    minimum: float | dict[str, float] | None,
) -> str | None:
    if minimum is None:
        return None
    thresholds = {"reward": minimum} if isinstance(minimum, (int, float)) else minimum
    for key, threshold in thresholds.items():
        actual = rewards.get(key, float("-inf")) if rewards else float("-inf")
        if actual < threshold:
            return f"{key}={actual} below min_reward {threshold}"
    return None


class HarborTaskset(Taskset[HarborTask, HarborConfig]):
    def load(self) -> Iterator[HarborTask]:
        root = dataset_dir(self.config)
        task_dirs = [
            toml_path.parent
            for toml_path in sorted(root.rglob("task.toml"))
            if (
                (toml_path.parent / "instruction.md").is_file()
                or (toml_path.parent / "steps").is_dir()
            )
            and (
                self.config.tasks is None or toml_path.parent.name in self.config.tasks
            )
        ]
        if not task_dirs:
            raise ValueError(f"no harbor tasks found in {root}")
        for idx, task_dir in enumerate(task_dirs):
            yield HarborTask(parse_task(task_dir, idx, self.config), self.config.task)
