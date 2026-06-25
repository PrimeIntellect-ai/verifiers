"""harbor: run a Harbor (Terminal-Bench) dataset.

`dataset` is a Harbor Hub package id (e.g. "org/name" or "org/name@ref"),
downloaded directly from the registry and cached on first use, or a local path to
Harbor task directories. Each task dir ships task.toml plus either a single root
instruction/tests pair or a Harbor multi-step `steps/` layout. Defaults to the
registry `harbor/hello-world` dataset.

The harness runs in a container and edits the task's declared workdir; then the task's
verifier (tests/test.sh, or each step's tests/test.sh) runs in the SAME container and the
reward it writes to /logs/verifier/reward.txt or reward.json becomes the score.
The verification lives on the taskset, so a harbor task runs under any harness.

A task's declared [environment].docker_image becomes a first-class `Task.image`
the Environment injects into the runtime (docker/prime both pull it). A task whose
environment is only a `Dockerfile` uses the harness runtime image by default.
`dockerfile_policy` can switch that behavior to a deterministic local Docker build or a
hard error. A task with no environment at all runs on the runtime image, unless
`require_image`.
"""

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import Field, model_validator

from verifiers.harbor import (
    HarborDockerfileEnvironment,
    HarborEnvironment,
    HarborImageEnvironment,
    HarborStep,
    HarborStepResult,
    RewardStrategy,
    harbor_step_summary,
    harbor_task_prompt,
    load_harbor_config,
    load_harbor_environment,
    load_harbor_steps,
    make_harbor_tar,
    parse_harbor_rewards,
    run_harbor_steps,
    scaled_timeout,
    sum_timeouts,
    valid_harbor_task_dir,
)
from verifiers.v1.decorators import reward
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import StrictBaseModel

CACHE = Path.home() / ".cache" / "harbor"
HARBOR_SUPABASE_URL = "https://ofhuhcpkvzjlejydnvyd.supabase.co"
HARBOR_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_Z-vuQbpvpG-PStjbh4yE0Q_e-d3MTIH"
TESTS_TAR = "/tmp/harbor_tests.tgz"
logger = logging.getLogger("verifiers.v1.tasksets.harbor")

DockerfilePolicy = Literal["build", "ignore", "error"]


class HarborConfig(TasksetConfig):
    dataset: str = "harbor/hello-world"
    """A Harbor Hub package id ("org/name" or "org/name@ref"), where ref is a
    tag, integer revision, or sha256 digest; or a local directory of Harbor task dirs."""
    tasks: list[str] | None = None
    """Optional subset of task names to load (None = all)."""
    timeout_multiplier: float = Field(1.0, gt=0)
    """Scale each task's agent and verifier timeouts."""
    resource_multiplier: float = Field(1.0, gt=0)
    """Scale each task's CPU, memory, and disk requests. GPU requests are unchanged."""
    require_image: bool = False
    """For a task with NO declared environment at all (no docker_image, no Dockerfile),
    whether to reject it (True) or run it on the runtime's default image (False)."""
    dockerfile_policy: DockerfilePolicy = "ignore"
    """How to handle a task whose environment is declared by `environment/Dockerfile` but not
    `[environment].docker_image`: ignore the Dockerfile and use the runtime image (default),
    build a local deterministic Docker image, or raise an error."""
    ignore_dockerfile: bool = False
    """Legacy alias for `dockerfile_policy = "ignore"`. Explicit `dockerfile_policy` wins when
    both are set."""

    @model_validator(mode="before")
    @classmethod
    def _legacy_ignore_dockerfile(cls, data):
        if (
            isinstance(data, dict)
            and data.get("ignore_dockerfile")
            and "dockerfile_policy" not in data
        ):
            data = dict(data)
            data["dockerfile_policy"] = "ignore"
        return data


class Author(StrictBaseModel):
    name: str | None = None
    email: str | None = None


class HarborTask(Task):
    """A Harbor task. The base fields carry instruction text (`prompt`), the
    resolved container `image`, total task timeouts/resources, and [task]
    name/description. `steps` is populated for Harbor multi-step tasks."""

    keywords: list[str] = Field(default_factory=list)
    authors: list[Author] = Field(default_factory=list)
    difficulty: str | None = None
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    task_dir: str = Field(exclude=True)
    """Host path to the task dir; used to stage tests/ to verify, not serialized."""
    environment: HarborEnvironment = Field(exclude=True)
    """Parsed Harbor environment spec, kept for diagnostics and policy-specific tests."""
    multi_step_reward_strategy: RewardStrategy = "mean"
    steps: list[HarborStep] = Field(default_factory=list)


def dataset_dir(dataset: str) -> Path:
    """Resolve a local Harbor dataset path or download a Harbor Hub package.

    Harbor refs are tags, integer revisions, or ``sha256:`` digests.
    """
    local = Path(dataset).expanduser()
    if local.is_dir():
        return local
    out = CACHE / dataset.replace("/", "_").replace("@", "_")
    if out.is_dir():
        return out

    package, _, ref = dataset.partition("@")
    org, name = package.split("/", 1)
    ref = ref or "latest"
    url = os.getenv("HARBOR_SUPABASE_URL", HARBOR_SUPABASE_URL).rstrip("/")
    key = os.getenv("HARBOR_SUPABASE_PUBLISHABLE_KEY", HARBOR_SUPABASE_PUBLISHABLE_KEY)
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    # Harbor stores datasets in version tables; standalone tasks use the RPC fallback below.
    version_query = {
        "package.name": f"eq.{name}",
        "package.type": "eq.dataset",
        "package.org.name": f"eq.{org}",
        "limit": 1,
    }
    if ref.isdigit():
        table = "dataset_version"
        version_query |= {
            "select": "id,package:package_id!inner(name,type,org:org_id!inner(name))",
            "revision": f"eq.{ref}",
        }
    elif ref.startswith("sha256:"):
        table = "dataset_version"
        version_query |= {
            "select": "id,package:package_id!inner(name,type,org:org_id!inner(name))",
            "content_hash": f"eq.{ref.removeprefix('sha256:')}",
        }
    else:
        table = "dataset_version_tag"
        version_query |= {
            "select": "dataset_version:dataset_version_id(id),package:package_id!inner(name,type,org:org_id!inner(name))",
            "tag": f"eq.{ref}",
        }
    request = Request(
        f"{url}/rest/v1/{table}?{urlencode(version_query)}", headers=headers
    )
    with urlopen(request) as response:
        versions = json.load(response)

    if versions:
        version = versions[0]
        version_id = version.get("id") or version["dataset_version"]["id"]
        task_query = {
            "select": "task_version:task_version_id(archive_path,package:package_id(name))",
            "dataset_version_id": f"eq.{version_id}",
            "order": "task_version_id",
            "limit": 1000,
            "offset": 0,
        }
        request = Request(
            f"{url}/rest/v1/dataset_version_task?{urlencode(task_query)}",
            headers={**headers, "Prefer": "count=exact"},
        )
        with urlopen(request) as response:
            pages = [json.load(response)]
            total = int(response.headers["Content-Range"].rsplit("/", 1)[1])

        for offset in range(1000, total, 1000):
            task_query["offset"] = offset
            request = Request(
                f"{url}/rest/v1/dataset_version_task?{urlencode(task_query)}",
                headers=headers,
            )
            with urlopen(request) as response:
                pages.append(json.load(response))
        downloads = [
            (
                Path(name) / row["task_version"]["package"]["name"],
                row["task_version"]["archive_path"],
            )
            for page in pages
            for row in page
        ]
    else:
        body = json.dumps({"p_org": org, "p_name": name, "p_ref": ref}).encode()
        request = Request(
            f"{url}/rest/v1/rpc/resolve_task_version", body, headers=headers
        )
        with urlopen(request) as response:
            task = json.load(response)
        if task is None:
            raise ValueError(f"Harbor package not found: {dataset}")
        downloads = [(Path(name), task["archive_path"])]

    if not downloads:
        raise ValueError(f"Harbor dataset has no tasks: {dataset}")

    CACHE.mkdir(parents=True, exist_ok=True)
    # Stream archives concurrently into a temporary directory; publish only a complete cache.
    with tempfile.TemporaryDirectory(dir=CACHE) as temp:

        def extract(item: tuple[Path, str]) -> None:
            relative_path, archive_path = item
            target = Path(temp) / relative_path
            target.mkdir(parents=True, exist_ok=True)
            request = Request(
                f"{url}/storage/v1/object/authenticated/packages/{archive_path}",
                headers=headers,
            )
            with urlopen(request) as response:
                with tarfile.open(fileobj=response, mode="r|gz") as tar:
                    kwargs = (
                        {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                    )
                    tar.extractall(target, **kwargs)

        with ThreadPoolExecutor(max_workers=100) as pool:
            list(pool.map(extract, downloads))
        try:
            Path(temp).rename(out)
        except OSError:
            if out.is_dir():
                return out
            raise
    return out


def dockerfile_context_digest(context: Path) -> str:
    """Content hash for a Docker build context, independent of mtimes."""
    digest = hashlib.sha256()
    for path in sorted(context.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(context).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def docker_tag_component(value: str) -> str:
    component = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip(".-")
    return component or "task"


def docker_image_exists(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def build_dockerfile_image(
    environment: HarborDockerfileEnvironment, task_name: str
) -> str:
    """Build a Harbor `environment/Dockerfile` into a deterministic local image tag."""
    context = environment.context
    dockerfile = context / "Dockerfile"
    if not dockerfile.is_file():
        raise FileNotFoundError(dockerfile)
    digest = dockerfile_context_digest(context)[:16]
    tag = f"vf-harbor-{docker_tag_component(task_name)}:{digest}"
    docker_bin = shutil.which("docker")
    if docker_bin is None:
        raise RuntimeError(
            f"{task_name}: Dockerfile environment requires the `docker` CLI to build. "
            "Install Docker, publish the image and set [environment].docker_image, or pass "
            "--taskset.dockerfile-policy ignore to run on the harness runtime image."
        )
    if docker_image_exists(tag):
        logger.info("harbor: using cached Dockerfile image %s", tag)
        return tag
    logger.info("harbor: building Dockerfile image %s from %s", tag, context)
    try:
        result = subprocess.run(
            [docker_bin, "build", "-t", tag, str(context)],
            capture_output=True,
            text=True,
            timeout=environment.build_timeout,
        )
    except subprocess.TimeoutExpired as e:
        timeout = environment.build_timeout
        detail = f" after {timeout:.0f}s" if timeout is not None else ""
        raise RuntimeError(f"{task_name}: Dockerfile build timed out{detail}") from e
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(f"{task_name}: Dockerfile build failed: {detail}")
    return tag


def resolve_image(
    task_dir: Path, environment: HarborEnvironment, harbor_config: HarborConfig
) -> str | None:
    """Resolve the task image within Harbor loading, keeping Dockerfile policy local to Harbor."""
    if isinstance(environment, HarborImageEnvironment):
        return environment.image
    if isinstance(environment, HarborDockerfileEnvironment):
        if harbor_config.dockerfile_policy == "ignore":
            return None
        if harbor_config.dockerfile_policy == "error":
            raise ValueError(
                f"{task_dir.name}: environment is a Dockerfile, not a pullable "
                "[environment].docker_image. Set --taskset.dockerfile-policy build to build "
                "it locally, publish an image and set [environment].docker_image, or pass "
                "--taskset.dockerfile-policy ignore to run on the harness runtime image."
            )
        return build_dockerfile_image(environment, task_dir.name)
    if harbor_config.require_image:
        raise ValueError(
            f"{task_dir.name}: no [environment].docker_image and require_image=True"
        )
    return None


def parse_resources(env: dict, multiplier: float = 1.0) -> TaskResources:
    """Map a task.toml [environment] block to TaskResources (0 gpus -> unset)."""
    return TaskResources(
        cpu=env["cpus"] * multiplier if env.get("cpus") else None,
        memory=env["memory_mb"] / 1024 * multiplier if env.get("memory_mb") else None,
        gpu=str(env["gpus"]) if env.get("gpus") else None,
        disk=env["storage_mb"] / 1024 * multiplier if env.get("storage_mb") else None,
    )


def parse_task(task_dir: Path, idx: int, harbor_config: HarborConfig) -> HarborTask:
    """Read a Harbor task dir into a typed task, supporting both single-step
    (`instruction.md` + `tests/`) and multi-step (`[[steps]]` + `steps/<name>/`) layouts."""
    config = load_harbor_config(task_dir)
    task, meta = config.get("task", {}), config.get("metadata", {})
    authors = [Author(**a) for a in task.get("authors", [])]
    if not authors and meta.get("author_name"):
        authors = [Author(name=meta["author_name"], email=meta.get("author_email"))]
    steps = load_harbor_steps(task_dir, config, harbor_config.timeout_multiplier)
    environment = load_harbor_environment(
        task_dir, config, harbor_config.timeout_multiplier
    )
    harness_timeout = scaled_timeout(
        config.get("agent", {}).get("timeout_sec"), harbor_config.timeout_multiplier
    )
    scoring_timeout = scaled_timeout(
        config.get("verifier", {}).get("timeout_sec"), harbor_config.timeout_multiplier
    )
    if steps:
        harness_timeout = sum_timeouts(step.harness_timeout for step in steps)
        scoring_timeout = sum_timeouts(step.scoring_timeout for step in steps)
    return HarborTask(
        idx=idx,
        name=task.get("name") or task_dir.name,
        description=task.get("description"),
        prompt=harbor_task_prompt(task_dir, steps),
        image=resolve_image(task_dir, environment, harbor_config),
        workdir=config.get("environment", {}).get("workdir"),
        timeout=TaskTimeout(harness=harness_timeout, scoring=scoring_timeout),
        resources=parse_resources(
            config.get("environment", {}), harbor_config.resource_multiplier
        ),
        keywords=task.get("keywords", []),
        authors=authors,
        difficulty=meta.get("difficulty"),
        category=meta.get("category"),
        tags=meta.get("tags", []),
        task_dir=str(task_dir),
        environment=environment,
        multi_step_reward_strategy=config.get("multi_step_reward_strategy") or "mean",
        steps=steps,
    )


class HarborTaskset(Taskset[HarborTask, HarborConfig]):
    def load_tasks(self) -> list[HarborTask]:
        root = dataset_dir(self.config.dataset)
        task_dirs = [
            toml_path.parent
            for toml_path in sorted(root.rglob("task.toml"))
            if valid_harbor_task_dir(toml_path.parent)
            and (
                self.config.tasks is None or toml_path.parent.name in self.config.tasks
            )
        ]
        if not task_dirs:
            raise ValueError(f"no harbor tasks found in {root}")
        return [
            parse_task(task_dir, idx, self.config)
            for idx, task_dir in enumerate(task_dirs)
        ]

    def record_step_results(
        self, trace: Trace, task: HarborTask, results: list[HarborStepResult]
    ) -> None:
        steps, aggregate = harbor_step_summary(results, task.multi_step_reward_strategy)
        trace.info["harbor_steps"] = steps
        trace.info["harbor_multi_step_reward"] = aggregate

    async def run_verifier(
        self,
        runtime: Runtime,
        tests_dir: Path,
        solution_dir: Path,
        *,
        shared_tests_dir: Path | None = None,
        verifier_env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, float]:
        directories = []
        if shared_tests_dir is not None:
            directories.append((shared_tests_dir, "tests"))
        directories.extend([(tests_dir, "tests"), (solution_dir, "oracle")])
        await runtime.write(TESTS_TAR, make_harbor_tar(directories))
        await runtime.run(
            [
                "sh",
                "-c",
                "rm -rf /tests /logs/verifier; "
                f"mkdir -p /tests /oracle /logs/verifier && tar -xzf {TESTS_TAR} -C /",
            ],
            {},
        )
        command = "cd /tests && bash test.sh"
        if timeout is not None:
            command = f"cd /tests && timeout {shlex.quote(f'{timeout}s')} bash test.sh"
        await runtime.run(["sh", "-c", command], verifier_env or {})
        return parse_harbor_rewards(await self.read_reward(runtime))

    async def read_reward(self, runtime: Runtime) -> str:
        for path in ("/logs/verifier/reward.txt", "/logs/verifier/reward.json"):
            try:
                return (await runtime.read(path)).decode().strip()
            except (SandboxError, OSError, RuntimeError, FileNotFoundError):
                continue
        return ""

    @reward(weight=1.0)
    async def solved(self, task: HarborTask, trace: Trace, runtime: Runtime) -> float:
        if task.steps:

            async def run_step(step: HarborStep) -> dict[str, float]:
                return await self.run_verifier(
                    runtime,
                    tests_dir=step.task_dir / "tests",
                    solution_dir=step.task_dir / "solution",
                    shared_tests_dir=Path(task.task_dir) / "tests",
                    verifier_env=step.verifier_env,
                    timeout=step.scoring_timeout,
                )

            results = await run_harbor_steps(task.steps, run_step)
            self.record_step_results(trace, task, results)
            rewards = trace.info["harbor_multi_step_reward"]
            return float(rewards.get("reward", 0.0))
        rewards = await self.run_verifier(
            runtime,
            tests_dir=Path(task.task_dir) / "tests",
            solution_dir=Path(task.task_dir) / "solution",
        )
        return float(rewards.get("reward", 0.0))
