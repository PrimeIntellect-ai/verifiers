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
import hashlib
import io
import logging
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated

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
    env: dict[str, str] = Field(default_factory=dict)
    healthcheck: dict | None = None
    fresh_copy: bool = False
    """Whether this came from Harbor's fresh copy of `[environment]` rather than a
    declared `[verifier.environment]`. A fresh copy inherits the agent box's resolved
    resources; a declared environment states its own, and what it omits falls back to
    the run's rather than to the agent's task-derived values."""
    network_allow: list[str] = Field(default_factory=lambda: ["*"])
    """Destinations the verifier may reach, from the verifier's network mode. `["*"]`
    is unrestricted; `[]` is Harbor's `no-network` / `allow_internet = false`."""


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


class HarborTask(Task[HarborData]):
    """Stage and run Harbor's verifier inside the task's live runtime."""

    verifier_elsewhere: bool = False
    verifier_staged: bool = False

    def runtime_env(self) -> dict[str, str]:
        return resolve_env(self.data.env)

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
        if self.data.healthcheck is not None:
            await self.wait_for_health(runtime, self.data.healthcheck)

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

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        """Run Harbor's collect hooks while the agent's box is still alive.

        Harbor runs these after the agent phase and before artifact collection, which
        is exactly what `finalize` means here, so the hook maps onto the existing
        lifecycle rather than needing a stage of its own.

        Strict, unlike `harbor run`, which logs a failed hook and carries on: there the
        output is observability, here it is a grading input, and a silently absent file
        makes the verifier score a stale state instead of failing loudly.
        """
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

    async def stage_tests(self, runtime: Runtime, wipe: bool = False) -> None:
        """Put the task package's `tests/` in `/tests`, where `test.sh` expects it.

        Raises rather than scoring stale state: a leftover reward file — planted by
        the agent or shipped in the image — must be gone before `test.sh` runs, so a
        removal that fails must not fall through to reading it.

        `wipe` for a box we did not watch being built: a fresh container of the task's
        image can ship its own `/tests`, and a leftover file there would be graded as
        though it came from the package.
        """
        await runtime.write(
            "/tmp/tests.tgz", make_tar(Path(self.data.task_dir) / "tests")
        )
        stage = (
            f"{'rm -rf /tests && ' if wipe else ''}"
            "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt && "
            "mkdir -p /logs/verifier /tests && tar -xzf /tmp/tests.tgz -C /tests"
        )
        result = await runtime.run(["sh", "-c", stage], {})
        if result.exit_code:
            raise TaskError(
                f"staging tests failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )

    @reward(weight=1.0)
    async def solved(self, runtime: Runtime, trace: Trace) -> float | dict[str, float]:
        if self.data.verifier is not None:
            if self.verifier_elsewhere:
                return {}
            if not self.verifier_staged:
                raise TaskError(
                    f"task {self.data.name!r} declares a separate verifier "
                    '([verifier].environment_mode = "separate"); grade it through '
                    "the harbor env (this taskset's default), or force shared "
                    "grading with --taskset.ignore-separate-verifier"
                )
        else:
            await self.stage_tests(runtime)
        return await self.run_verifier(runtime, trace)

    async def run_verifier(
        self, runtime: Runtime, trace: Trace
    ) -> float | dict[str, float]:
        # By absolute path, in the runtime's configured workdir: Harbor execs the
        # script the same way, and scripts do grade the agent's work at `$PWD`.
        await runtime.run(
            ["bash", "/tests/test.sh"], resolve_env(self.data.verifier_env)
        )
        scores = await self.read_reward_json(runtime)
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

    async def read_reward_json(
        self, runtime: Runtime
    ) -> float | dict[str, float] | None:
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
            "upload_environment": data.upload_environment if fresh else False,
            "env": dict(verifier.env),
            "healthcheck": verifier.healthcheck,
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
        prompt=harbor_task.instruction.strip(),
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
        **environment.model_dump(include={"env", "healthcheck"}, mode="json"),
        verifier_env=parsed.verifier.env,
        artifacts=artifacts,
        collect=hooks,
        verifier=verifier,
    )


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
    from harbor.constants import MAIN_SERVICE_NAME
    from harbor.models.task.artifacts import (
        effective_artifact_service,
        normalize_artifact_entries,
    )

    verifier = parsed.verifier
    if verifier.user is not None:
        raise ValueError(f"{task_dir.name}: [verifier].user is not supported")

    artifacts: list[Artifact] = []
    for entry in normalize_artifact_entries(parsed.artifacts):
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

    hooks: list[CollectHook] = []
    for hook in verifier.collect:
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

    return artifacts, hooks, parse_verifier_environment(task_dir, parsed, harbor_config)


def parse_verifier_environment(
    task_dir: Path, parsed, harbor_config: HarborConfig
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
        resolve_task_verifier_mode,
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

    environment = resolve_effective_verifier_env_config(parsed, None)
    if environment is None:  # unreachable while the mode is SEPARATE
        raise ValueError(f"{task_dir.name}: separate verifier resolved no environment")
    declared = parsed.verifier.environment is not None

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
        for field in ("mcp_servers", "skills_dir", "gpu_types", "tpu")
        if getattr(environment, field, None)
    ]
    if environment.os != TaskOS.LINUX or unsupported:
        raise ValueError(
            f"{task_dir.name}: verifier environment declares "
            f"{unsupported or environment.os}, which verifiers' verifier-runtime "
            "integration cannot honor"
        )

    network = parsed.verifier.explicit_phase_policy() or environment.resolve_baseline()
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
        **environment.model_dump(include={"env", "healthcheck"}, mode="json"),
        fresh_copy=not declared,
        network_allow=(
            ["*"]
            if network.network_mode == NetworkMode.PUBLIC
            else list(network.allowed_hosts)
        ),
    )


def resolve_env(values: dict[str, str]) -> dict[str, str]:
    """Resolve templates at execution time so host secrets are never serialized."""
    if not values:
        return {}

    # Harbor is an optional dependency, so importing this module must still work
    # for users who do not install the Harbor extra.
    from harbor.utils.env import resolve_env_vars

    return resolve_env_vars(values)


# Downloaded task directories are immutable. Cache the current task's environment
# and tests to bound memory while reusing both archives across rollouts.
@lru_cache(maxsize=2)
def make_tar(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name)
    return buffer.getvalue()


class HarborTaskset(Taskset[HarborTask, HarborConfig]):
    def load(self) -> Iterator[HarborTask]:
        root = dataset_dir(self.config)
        task_dirs = [
            toml_path.parent
            for toml_path in sorted(root.rglob("task.toml"))
            if (toml_path.parent / "instruction.md").is_file()
            and (
                self.config.tasks is None or toml_path.parent.name in self.config.tasks
            )
        ]
        if not task_dirs:
            raise ValueError(f"no harbor tasks found in {root}")
        for idx, task_dir in enumerate(task_dirs):
            yield HarborTask(parse_task(task_dir, idx, self.config), self.config.task)
