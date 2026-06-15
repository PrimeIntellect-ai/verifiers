"""harbor: run a Harbor (Terminal-Bench) dataset.

`dataset` is a Harbor Hub registry id (e.g. "name", "name@version",
"org/name@ref"), downloaded + cached on first use via the `harbor` CLI
(`uv tool install harbor`). Each task dir ships task.toml + instruction.md
(+ tests/, solution/, environment/). Defaults to the registry `hello-world` task.

The harness runs in the task's container and edits its filesystem. By default,
tests/test.sh verifies that same container. Harbor's separate verifier mode starts
a fresh runtime from [verifier.environment], copies only /logs/artifacts plus the
task's configured artifacts, and runs /tests/test.sh there. Explicit verifier
environments need a pullable image that owns the tests; bare separate mode reuses
the task image and uploads tests because V1 does not build tests/Dockerfile.
Verification lives on the taskset (the `solved` reward), so a Harbor task runs
under any harness.

A task's declared [environment].docker_image becomes a first-class `Task.image`
the Environment injects into the runtime (docker/prime both pull it). Tasks that
only ship an environment/Dockerfile have no pullable image: with `require_image`
they're rejected; otherwise they run on the runtime's default image (we don't
build the Dockerfile — a locally-built image isn't pullable by a remote sandbox).
"""

import io
import json
import os
import re
import subprocess
import tarfile
import tomllib
from pathlib import Path

from pydantic import Field

from verifiers.v1.decorators import reward
from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import Runtime, SubprocessConfig, make_runtime
from verifiers.v1.task import Resources, Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.types import StrictBaseModel

CACHE = Path.home() / ".cache" / "harbor"
ENV_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


class HarborConfig(TasksetConfig):
    dataset: str = "hello-world"
    """A Harbor Hub registry id ("name", "name@version", "org/name@ref"),
    downloaded + cached via the `harbor` CLI."""
    tasks: list[str] | None = None
    """Optional subset of task names to load (None = all)."""
    require_image: bool = False
    """For a task with NO declared environment at all (no docker_image, no Dockerfile),
    whether to reject it (True) or run it on the runtime's default image (False). Tasks
    whose environment is a `Dockerfile` are always rejected — building Dockerfiles isn't
    supported, and running them on the default image scores against the wrong env."""


class Author(StrictBaseModel):
    name: str | None = None
    email: str | None = None


class HarborVerifier(StrictBaseModel):
    image: str | None = None
    workdir: str | None = None
    resources: Resources = Resources()
    network_access: bool = True
    env: dict[str, str] = {}
    artifacts: list[str] = []
    upload_tests: bool = False


class HarborTask(Task):
    """A Harbor task. The base fields carry instruction.md (`instruction`), the
    resolved container `image`, the `harness_timeout`/`scoring_timeout`/`resources`
    (from task.toml's [harness]/[verifier]/[environment]), and [task].name/description;
    the rest mirror [metadata]."""

    keywords: list[str] = []
    authors: list[Author] = []
    difficulty: str | None = None
    category: str | None = None
    tags: list[str] = []
    task_dir: str = Field(exclude=True)
    """Host path to the task dir; used to stage tests/ to verify, not serialized."""
    verifier: HarborVerifier = Field(exclude=True)
    """Resolved Harbor verifier mode and environment, used only while scoring."""


def dataset_dir(dataset: str) -> Path:
    """Download a Harbor Hub `dataset` to a directory of task dirs via the `harbor`
    CLI, cached on first use."""
    out = CACHE / dataset.replace("/", "_").replace("@", "_")
    if not out.is_dir():
        try:
            subprocess.run(
                ["harbor", "download", dataset, "--export", "-o", str(out)], check=True
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"the `harbor` CLI is needed to download {dataset!r}; "
                "install it with `uv tool install harbor`"
            ) from e
    return out


def resolve_image(task_dir: Path, config: dict, require_image: bool) -> str | None:
    """The task's declared registry image (usable by docker or prime). A pullable
    `[environment].docker_image` is used directly. A task whose environment is a
    `Dockerfile` is rejected — we don't build Dockerfiles, and running it on the default
    image would silently score against the wrong environment (e.g. SWE-bench's `/testbed`
    repo would be missing). A task with no environment at all runs on the runtime's
    default image, unless `require_image`."""
    declared = config.get("environment", {}).get("docker_image")
    if declared:
        return declared
    if (task_dir / "environment" / "Dockerfile").exists():
        raise ValueError(
            f"{task_dir.name}: environment is a Dockerfile, not a pullable "
            "[environment].docker_image — building Dockerfiles isn't supported, so this "
            "task can't run (it would otherwise score against the wrong default image)."
        )
    if require_image:
        raise ValueError(
            f"{task_dir.name}: no [environment].docker_image and require_image=True"
        )
    return None


def parse_resources(env: dict) -> Resources:
    """Map a task.toml [environment] block to Resources (0 gpus -> unset)."""
    return Resources(
        cpu=env.get("cpus"),
        memory=env["memory_mb"] / 1024 if env.get("memory_mb") else None,
        gpu=str(env["gpus"]) if env.get("gpus") else None,
        disk=env["storage_mb"] / 1024 if env.get("storage_mb") else None,
    )


def parse_task(task_dir: Path, idx: int, require_image: bool) -> HarborTask:
    """Read a harbor task dir (task.toml + instruction.md) into a typed task,
    handling both the [task].authors and legacy [metadata].author_name layouts."""
    config = tomllib.loads((task_dir / "task.toml").read_text())
    task, meta = config.get("task", {}), config.get("metadata", {})
    environment = config.get("environment", {})
    verifier_config = config.get("verifier", {})
    verifier_environment = verifier_config.get("environment")
    separate = (
        verifier_config.get("environment_mode") == "separate"
        or verifier_environment is not None
    )
    verifier_environment = (
        environment if verifier_environment is None else verifier_environment
    )
    if separate and not verifier_environment.get("docker_image"):
        raise ValueError(
            f"{task_dir.name}: separate verification needs a pullable docker_image; "
            "building tests/Dockerfile is not supported"
        )
    network_mode = verifier_config.get(
        "network_mode", verifier_environment.get("network_mode", "public")
    )
    artifacts = [
        artifact if isinstance(artifact, str) else artifact["source"]
        for artifact in config.get("artifacts", [])
    ]

    authors = [Author(**a) for a in task.get("authors", [])]
    if not authors and meta.get("author_name"):
        authors = [Author(name=meta["author_name"], email=meta.get("author_email"))]
    return HarborTask(
        idx=idx,
        name=task.get("name") or task_dir.name,
        description=task.get("description"),
        instruction=(task_dir / "instruction.md").read_text().strip(),
        image=resolve_image(task_dir, config, require_image),
        harness_timeout=config.get("agent", {}).get("timeout_sec"),
        scoring_timeout=verifier_config.get("timeout_sec"),
        resources=parse_resources(environment),
        keywords=task.get("keywords", []),
        authors=authors,
        difficulty=meta.get("difficulty"),
        category=meta.get("category"),
        tags=meta.get("tags", []),
        task_dir=str(task_dir),
        verifier=HarborVerifier(
            image=verifier_environment.get("docker_image") if separate else None,
            workdir=verifier_environment.get("workdir") if separate else None,
            resources=parse_resources(verifier_environment)
            if separate
            else Resources(),
            network_access=network_mode == "public",
            env={
                str(key): str(value)
                for key, value in verifier_config.get("env", {}).items()
            },
            artifacts=artifacts,
            upload_tests=separate and verifier_config.get("environment") is None,
        ),
    )


def make_tar(directory: Path) -> bytes:
    """Tar a directory's contents (flat) into a gzipped tarball."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for item in sorted(directory.iterdir()):
            tar.add(item, arcname=item.name)
    return buffer.getvalue()


class HarborTaskset(Taskset[HarborTask, HarborConfig]):
    def load_tasks(self) -> list[HarborTask]:
        root = dataset_dir(self.config.dataset)
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
        return [
            parse_task(task_dir, idx, self.config.require_image)
            for idx, task_dir in enumerate(task_dirs)
        ]

    @reward(weight=1.0)
    async def solved(self, task: HarborTask, runtime: Runtime) -> float:
        target = runtime
        verifier_runtime: Runtime | None = None
        verifier = task.verifier
        archive = "/tmp/vf-harbor-artifacts.tgz"
        archive_data = b""
        if verifier.image is not None:
            runtime_config = runtime.config
            if isinstance(runtime_config, SubprocessConfig):
                raise ProgramError(
                    "separate Harbor verification needs a docker, prime, or modal runtime"
                )
            result = await runtime.run(["mkdir", "-p", "/logs/artifacts"], {})
            if result.exit_code:
                raise ProgramError(
                    f"Harbor artifact directory setup failed: "
                    f"{result.stderr or result.stdout}"
                )
            result = await runtime.run(
                [
                    "tar",
                    "-czf",
                    archive,
                    *dict.fromkeys(["/logs/artifacts", *verifier.artifacts]),
                ],
                {},
            )
            if result.exit_code:
                raise ProgramError(
                    f"Harbor artifact collection failed: "
                    f"{result.stderr or result.stdout}"
                )
            archive_data = await runtime.read(archive)

            config_type = type(runtime_config)
            updates = {
                name: config_type.model_fields[name].default
                for name in Resources.model_fields
                if name in config_type.model_fields
            }
            updates.update(verifier.resources.model_dump(exclude_none=True))
            updates.update(
                image=verifier.image,
                network_access=verifier.network_access,
            )
            if verifier.workdir is not None:
                updates["workdir"] = verifier.workdir
            verifier_runtime = make_runtime(
                runtime_config.model_copy(update=updates),
                name=f"{runtime.name}-verifier",
            )

        try:
            if verifier_runtime is not None:
                await verifier_runtime.start()
                target = verifier_runtime
                await target.write(archive, archive_data)
                result = await target.run(["tar", "-xzf", archive, "-C", "/"], {})
                if result.exit_code:
                    raise ProgramError(
                        f"Harbor artifact extraction failed: "
                        f"{result.stderr or result.stdout}"
                    )

            result = await target.run(
                ["sh", "-c", "mkdir -p /logs/verifier /logs/artifacts"], {}
            )
            if result.exit_code:
                raise ProgramError(
                    f"Harbor verifier directory setup failed: "
                    f"{result.stderr or result.stdout}"
                )
            if verifier.image is None or verifier.upload_tests:
                await target.write(
                    "/tmp/tests.tgz", make_tar(Path(task.task_dir) / "tests")
                )
                result = await target.run(
                    [
                        "sh",
                        "-c",
                        "mkdir -p /logs/verifier /tests && "
                        "tar -xzf /tmp/tests.tgz -C /tests",
                    ],
                    {},
                )
                if result.exit_code:
                    raise ProgramError(
                        f"Harbor test upload failed: {result.stderr or result.stdout}"
                    )

            env: dict[str, str] = {}
            for key, value in verifier.env.items():
                match = ENV_TEMPLATE.fullmatch(value)
                if match is not None:
                    name, default = match.groups()
                    value = os.environ.get(name, default)
                if value is None:
                    raise ProgramError(
                        f"environment variable {name!r} required by Harbor verifier is unset"
                    )
                env[key] = value

            await target.run(
                [
                    "sh",
                    "-c",
                    "cd /tests && bash test.sh > /logs/verifier/test-stdout.txt 2>&1",
                ],
                env,
            )
            result = await target.run(
                [
                    "sh",
                    "-c",
                    "if [ -s /logs/verifier/reward.txt ]; then "
                    "cat /logs/verifier/reward.txt; "
                    "elif [ -s /logs/verifier/reward.json ]; then "
                    "cat /logs/verifier/reward.json; fi",
                ],
                {},
            )
            if result.exit_code:
                raise ProgramError(
                    f"Harbor reward read failed: {result.stderr or result.stdout}"
                )
            output = (result.stdout or "").strip()
            try:
                return float(output or 0)
            except ValueError:
                try:
                    return float(json.loads(output).get("reward", 0))
                except (AttributeError, TypeError, ValueError):
                    return 0.0
        finally:
            if verifier_runtime is not None:
                await verifier_runtime.stop()


def load_taskset(config: HarborConfig) -> HarborTaskset:
    return HarborTaskset(config)
