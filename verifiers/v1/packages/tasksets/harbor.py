import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import cast

from verifiers.utils.import_utils import load_toml

from ...config import CallableEntry, TasksetConfig
from ...taskset import Taskset
from ...utils.sandbox_utils import SandboxClient, create_sandbox_lease
from verifiers.decorators import reward
from ...types import ConfigData, ConfigMap

TASKS_SUBDIR = "tasks"
VERIFIER_MODE_SHARED = "shared"
VERIFIER_MODE_SEPARATE = "separate"
HARBOR_ARTIFACTS_DIR = "/logs/artifacts"
HARBOR_REWARD_COMMAND = (
    "if [ -s /logs/verifier/reward.json ]; then "
    "cat /logs/verifier/reward.json; "
    "elif [ -s /logs/verifier/reward.txt ]; then "
    "cat /logs/verifier/reward.txt; fi"
)
ENV_TEMPLATE_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")


def _resolve_caller_package() -> str | None:
    for frame_info in inspect.stack()[1:]:
        package = frame_info.frame.f_globals.get("__package__")
        if not isinstance(package, str) or not package:
            package = frame_info.frame.f_globals.get("__name__")
        if not isinstance(package, str) or not package or package == "__main__":
            continue
        if package.startswith("verifiers"):
            continue
        return package
    return None


def _bundle_tasks_root(module_name: str) -> Path:
    try:
        tasks = cast(os.PathLike[str], files(module_name) / TASKS_SUBDIR)
        return Path(os.fspath(tasks))
    except TypeError as exc:
        module = sys.modules.get(module_name)
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str):
            raise exc
        return Path(module_file).resolve().parent / TASKS_SUBDIR


class HarborTasksetConfig(TasksetConfig):
    rewards: list[CallableEntry] = ["harbor_reward"]
    dataset: str | None = None
    bundle_package: str | None = None
    task_names: list[str] | None = None
    cache_dir: str | None = None
    refresh: bool = False
    docker_image: str = "python:3.11-slim"
    cpu_cores: float = 2.0
    memory_gb: float = 4.0
    disk_size_gb: float = 10.0
    timeout_minutes: int = 120
    agent_timeout_seconds: float = 900.0
    verifier_timeout_seconds: float = 900.0
    workdir: str = "/app"
    task_dir: str = "/task"
    scope: str = "rollout"
    env: dict[str, str] = {}


class HarborTaskset(Taskset[HarborTasksetConfig]):
    def __init__(self, config: HarborTasksetConfig | None = None):
        config = HarborTasksetConfig() if config is None else config
        assert isinstance(config, HarborTasksetConfig)
        if config.dataset is not None and not isinstance(config.dataset, str):
            raise TypeError("HarborTaskset dataset must be a string.")
        if config.dataset is None and config.bundle_package is None:
            config = config.model_copy(
                update={"bundle_package": _resolve_caller_package()}
            )
        cache_dir_value = config.cache_dir
        self._cache_dir = (
            Path(str(cache_dir_value)).expanduser() if cache_dir_value else None
        )
        if config.scope not in {"rollout", "group", "global"}:
            raise ValueError("HarborTaskset scope must be rollout, group, or global.")
        super().__init__(config=config)
        self.taskset_id = self.config.taskset_id or "harbor"

    @property
    def task_names(self) -> list[str]:
        return list(self.config.task_names or [])

    @property
    def cpu_cores(self) -> float:
        return self.config.cpu_cores

    def resolve_tasks_root(self) -> Path:
        return resolve_tasks_root(self.config, cache_dir=self._cache_dir)

    def load_tasks(self) -> list[ConfigData]:
        return load_tasks(self.config)


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_tasks(config: HarborTasksetConfig) -> list[ConfigData]:
    cache_dir = Path(str(config.cache_dir)).expanduser() if config.cache_dir else None
    root = resolve_tasks_root(config, cache_dir=cache_dir)
    task_dirs = harbor_task_dirs(root, list(config.task_names or []))
    rows = [
        harbor_task_row(config, task_dir, index)
        for index, task_dir in enumerate(task_dirs)
    ]
    if not rows:
        raise ValueError(f"No valid Harbor tasks found in {root}.")
    return rows


def resolve_tasks_root(
    config: HarborTasksetConfig, *, cache_dir: Path | None = None
) -> Path:
    if config.dataset is not None:
        return download_harbor_dataset(
            config.dataset,
            cache_dir=cache_dir,
            refresh=config.refresh,
        )
    if config.bundle_package is None:
        raise RuntimeError(
            "HarborTaskset() without a dataset must be constructed from inside "
            "an installed Python package. Pass dataset='...' to fetch from "
            "Harbor Hub, or construct it from a packaged environment."
        )
    root = _bundle_tasks_root(config.bundle_package)
    if not root.exists():
        raise FileNotFoundError(
            "HarborTaskset() without a dataset requires "
            f"{config.bundle_package}/{TASKS_SUBDIR}/ to contain Harbor task "
            f"directories. Not found: {root}"
        )
    return root


def harbor_task_row(
    config: HarborTasksetConfig, task_dir: Path, index: int
) -> ConfigData:
    task_toml_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    with task_toml_path.open("rb") as f:
        task_config = load_toml(f)
    environment = task_config.get("environment", {}) or {}
    if not isinstance(environment, Mapping):
        raise TypeError(f"{task_toml_path} [environment] must be a mapping.")
    agent_config = task_config.get("agent", {}) or {}
    verifier_config = task_config.get("verifier", {}) or {}
    if not isinstance(agent_config, Mapping):
        raise TypeError(f"{task_toml_path} [agent] must be a mapping.")
    if not isinstance(verifier_config, Mapping):
        raise TypeError(f"{task_toml_path} [verifier] must be a mapping.")
    instruction = instruction_path.read_text().strip()
    task_remote_dir = config.task_dir.rstrip("/") or "/task"
    test_timeout = parse_number(
        verifier_config.get("timeout_sec"),
        config.verifier_timeout_seconds,
    )
    verifier_environment = verifier_config.get("environment")
    verifier_mode = verifier_config.get("environment_mode")
    if verifier_mode is not None:
        verifier_mode = str(verifier_mode)
        if verifier_mode not in {VERIFIER_MODE_SHARED, VERIFIER_MODE_SEPARATE}:
            raise ValueError(
                f"{task_toml_path} [verifier].environment_mode must be "
                "'shared' or 'separate'."
            )
    elif verifier_environment is not None:
        verifier_mode = VERIFIER_MODE_SEPARATE
    else:
        verifier_mode = VERIFIER_MODE_SHARED
    if verifier_mode == VERIFIER_MODE_SHARED and verifier_environment is not None:
        raise ValueError(
            f"{task_toml_path} [verifier].environment_mode='shared' is "
            "incompatible with [verifier.environment]."
        )
    if (
        verifier_mode == VERIFIER_MODE_SEPARATE
        and verifier_environment is not None
        and not isinstance(verifier_environment, Mapping)
    ):
        raise TypeError(f"{task_toml_path} [verifier.environment] must be a mapping.")
    agent_timeout = int(
        parse_number(agent_config.get("timeout_sec"), config.agent_timeout_seconds)
    )
    sandbox = harbor_sandbox_config(config, environment, agent_timeout)
    verifier_sandbox: ConfigData | None = None
    verifier_upload_tests = False
    if verifier_mode == VERIFIER_MODE_SEPARATE and verifier_environment is None:
        verifier_sandbox = {**sandbox, "command_timeout": int(test_timeout)}
        verifier_upload_tests = True
    elif verifier_mode == VERIFIER_MODE_SEPARATE:
        verifier_sandbox = harbor_sandbox_config(
            config,
            cast(ConfigMap, verifier_environment),
            int(test_timeout),
        )
    return {
        "example_id": index,
        "task_name": task_dir.name,
        "instruction": instruction,
        "task_toml": task_toml_path.read_text(),
        "task_dir": str(task_dir),
        "prompt": [{"role": "user", "content": instruction}],
        "sandbox": sandbox,
        "program": {
            "files": {
                f"{task_remote_dir}/instruction.md": {"task": "instruction"},
                f"{task_remote_dir}/task.toml": {"task": "task_toml"},
            },
            "env": {
                "HARBOR_TASK_NAME": task_dir.name,
                "HARBOR_TASK_DIR": task_remote_dir,
                "HARBOR_INSTRUCTION_PATH": f"{task_remote_dir}/instruction.md",
                "AGENT_WORKDIR": config.workdir,
                **config.env,
            },
        },
        "harbor": {
            "task_dir": str(task_dir),
            "task_name": task_dir.name,
            "config": task_config,
            "docker_image": environment.get("docker_image"),
            "test_timeout": test_timeout,
            "verifier_mode": verifier_mode,
            "verifier_sandbox": verifier_sandbox,
            "verifier_upload_tests": verifier_upload_tests,
            "verifier_env": verifier_config.get("env") or {},
            "artifacts": task_config.get("artifacts") or [],
        },
        "info": {
            "harbor": {
                "task_name": task_dir.name,
                "docker_image": environment.get("docker_image"),
            }
        },
    }


def harbor_sandbox_config(
    config: HarborTasksetConfig, environment: ConfigMap, command_timeout: int
) -> ConfigData:
    memory = (
        f"{environment['memory_mb']}mb"
        if "memory_mb" in environment
        else environment.get("memory")
    )
    storage = (
        f"{environment['storage_mb']}mb"
        if "storage_mb" in environment
        else environment.get("storage")
    )
    sandbox: ConfigData = {
        "image": environment.get("docker_image") or config.docker_image,
        "cpu_cores": parse_number(environment.get("cpus"), config.cpu_cores),
        "memory_gb": parse_gb(memory, config.memory_gb),
        "disk_size_gb": parse_gb(storage, config.disk_size_gb),
        "timeout_minutes": config.timeout_minutes,
        "command_timeout": command_timeout,
        "workdir": config.workdir,
        "scope": config.scope,
    }
    if "allow_internet" in environment:
        sandbox["network_access"] = bool(environment["allow_internet"])
    return sandbox


def harbor_task_dirs(root: Path, task_names: Iterable[str] | None = None) -> list[Path]:
    selected = set(task_names or [])
    if not root.exists():
        raise FileNotFoundError(f"Harbor tasks path not found: {root}")
    tasks: list[Path] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            raise ValueError(
                f"Harbor tasks root {root} contains non-directory entry {task_dir}."
            )
        if not is_harbor_task_dir(task_dir):
            raise ValueError(
                f"Malformed Harbor task {task_dir}: missing task.toml or "
                "instruction.md."
            )
        if not selected or task_dir.name in selected:
            tasks.append(task_dir)
    if selected:
        found = {path.name for path in tasks}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"Requested Harbor tasks not found: {missing}.")
    return tasks


def is_harbor_task_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "task.toml").exists()
        and (path / "instruction.md").exists()
    )


def parse_number(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError("Expected a numeric value.")
    return float(value)


def parse_gb(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().lower()
    if text.endswith("gb"):
        return float(text[:-2])
    if text.endswith("g"):
        return float(text[:-1])
    if text.endswith("mb"):
        return float(text[:-2]) / 1024
    if text.endswith("m"):
        return float(text[:-1]) / 1024
    return float(text)


def download_harbor_dataset(
    dataset_id: str, *, cache_dir: Path | None = None, refresh: bool = False
) -> Path:
    harbor_bin = shutil.which("harbor")
    uvx_bin = shutil.which("uvx")
    if harbor_bin is None and uvx_bin is None:
        raise FileNotFoundError(
            f"Harbor dataset {dataset_id!r} requires the Harbor CLI or uvx. "
            "Install Harbor or uvx before using Harbor Hub datasets."
        )
    root = cache_dir or Path.home() / ".cache" / "verifiers" / "harbor"
    dataset_dir = root / safe_dataset_dir_name(dataset_id)
    task_root = dataset_dir / dataset_id.rsplit("/", 1)[-1]
    if dataset_dir.exists() and not refresh:
        return task_root
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    if harbor_bin is not None:
        command = [
            harbor_bin,
            "datasets",
            "download",
            dataset_id,
            "--output-dir",
            str(dataset_dir),
        ]
    else:
        assert uvx_bin is not None
        command = [
            uvx_bin,
            "harbor",
            "datasets",
            "download",
            dataset_id,
            "--output-dir",
            str(dataset_dir),
        ]
    if refresh:
        command.append("--overwrite")
    subprocess.run(command, check=True)
    return task_root


def safe_dataset_dir_name(dataset_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_id).strip("_") or "dataset"


@reward(weight=1.0)
async def harbor_reward(task, state) -> float:
    if state.get("error") is not None:
        return 0.0
    sandbox_id = state.get("sandbox_id")
    if not isinstance(sandbox_id, str):
        return 0.0
    harbor = task.get("harbor")
    if not isinstance(harbor, Mapping):
        return 0.0
    task_dir = Path(str(harbor["task_dir"]))
    mode = str(harbor.get("verifier_mode") or VERIFIER_MODE_SHARED)
    timeout = int(parse_number(harbor.get("test_timeout"), 900))
    verifier_env = harbor_verifier_env(harbor)
    from prime_sandboxes import AsyncSandboxClient

    client = cast(SandboxClient, AsyncSandboxClient())
    try:
        if mode == VERIFIER_MODE_SEPARATE:
            reward_text = await run_separate_harbor_verifier(
                client, sandbox_id, harbor, state, timeout, verifier_env
            )
        else:
            await upload_harbor_tests(client, sandbox_id, task_dir)
            reward_text = await run_harbor_tests(
                client,
                sandbox_id,
                state,
                command="bash test.sh",
                working_dir="/tests",
                timeout=timeout,
                env=verifier_env,
            )
    except Exception as e:
        state["harbor_error"] = str(e)
        return 0.0
    finally:
        await client.aclose()
    return parse_reward_text(str(reward_text or "").strip())


def harbor_verifier_env(harbor: ConfigMap) -> dict[str, str] | None:
    raw_env = harbor.get("verifier_env") or {}
    if not isinstance(raw_env, Mapping):
        raise TypeError("[verifier].env must be a mapping.")
    env: dict[str, str] = {}
    for key, value in raw_env.items():
        text = str(value)
        match = ENV_TEMPLATE_PATTERN.fullmatch(text)
        if match:
            var_name = match.group(1)
            default = match.group(2)
            if var_name in os.environ:
                text = os.environ[var_name]
            elif default is not None:
                text = default
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' not found in host environment"
                )
        env[str(key)] = text
    return env or None


async def run_separate_harbor_verifier(
    agent_client: SandboxClient,
    agent_sandbox_id: str,
    harbor: ConfigMap,
    state: ConfigData,
    timeout: int,
    verifier_env: dict[str, str] | None,
) -> str:
    """Run Harbor's separate verifier mode in a fresh sandbox.

    This is needed only when Harbor resolves the verifier environment separately
    from the agent environment; the verifier image owns /tests/test.sh, so we
    transfer just the configured grading inputs before running it.
    """
    sandbox = harbor.get("verifier_sandbox")
    if not isinstance(sandbox, Mapping):
        raise RuntimeError("Separate Harbor verifier did not resolve a sandbox.")
    lease = await create_sandbox_lease(cast(ConfigData, dict(sandbox)), "harbor")
    state["harbor_verifier_sandbox_id"] = lease.id
    try:
        await lease.execute("mkdir -p /logs/verifier /logs/artifacts /tests")
        if harbor.get("verifier_upload_tests"):
            await upload_harbor_tests(
                lease.client, lease.id, Path(str(harbor["task_dir"]))
            )
        await transfer_harbor_verifier_inputs(
            agent_client,
            agent_sandbox_id,
            lease.client,
            lease.id,
            harbor,
        )
        return await run_harbor_tests(
            lease.client,
            lease.id,
            state,
            command="bash test.sh",
            working_dir="/tests",
            timeout=timeout,
            env=verifier_env,
        )
    finally:
        await lease.delete()


async def run_harbor_tests(
    client: SandboxClient,
    sandbox_id: str,
    state: ConfigData,
    *,
    command: str,
    working_dir: str | None,
    timeout: int,
    env: dict[str, str] | None,
) -> str:
    result = await client.run_background_job(
        sandbox_id=sandbox_id,
        command=command,
        working_dir=working_dir,
        timeout=timeout,
        env=env,
    )
    state["harbor_tests"] = {
        "returncode": result.exit_code,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
    reward_result = await client.execute_command(
        sandbox_id=sandbox_id,
        command=HARBOR_REWARD_COMMAND,
    )
    return reward_result.stdout or ""


async def transfer_harbor_verifier_inputs(
    agent_client: SandboxClient,
    agent_sandbox_id: str,
    verifier_client: SandboxClient,
    verifier_sandbox_id: str,
    harbor: ConfigMap,
) -> None:
    raw_artifacts = harbor.get("artifacts") or []
    if not isinstance(raw_artifacts, list):
        raise TypeError("Harbor task artifacts must be a list.")
    artifacts: list[tuple[str, list[str]]] = []
    has_artifacts_dir = False
    for artifact in raw_artifacts:
        if isinstance(artifact, str):
            source = artifact
            exclude: list[str] = []
        elif isinstance(artifact, Mapping):
            artifact_data = cast(ConfigMap, artifact)
            source_value = artifact_data.get("source")
            if not isinstance(source_value, str):
                raise TypeError("Harbor artifacts must be strings or source mappings.")
            source = source_value
            raw_exclude = artifact_data.get("exclude") or []
            if not isinstance(raw_exclude, list):
                raise TypeError("Harbor artifact exclude must be a list.")
            exclude = [str(item) for item in raw_exclude]
        else:
            raise TypeError("Harbor artifacts must be strings or source mappings.")

        if source.rstrip("/") == HARBOR_ARTIFACTS_DIR:
            has_artifacts_dir = True
        artifacts.append((source, exclude))
    if not has_artifacts_dir:
        artifacts.insert(0, (HARBOR_ARTIFACTS_DIR, []))

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        local_tar = Path(tmp_file.name)
    remote_tar = "/tmp/_vf_harbor_inputs.tar.gz"
    archive_command_lines = [
        "set -e",
        f"rm -f {shlex.quote(remote_tar)}",
        "tmp=$(mktemp -d /tmp/_vf_harbor_inputs.XXXXXX)",
        "trap 'rm -rf \"$tmp\"' EXIT",
        "added=0",
    ]
    for source, exclude in artifacts:
        source = source.rstrip("/") or "/"
        path = source.lstrip("/")
        if not path:
            continue
        target = f'"$tmp"/{shlex.quote(path)}'
        parent = str(PurePosixPath(path).parent)
        mkdir_target = '"$tmp"' if parent == "." else f'"$tmp"/{shlex.quote(parent)}'
        exclude_args = " ".join(f"--exclude={shlex.quote(item)}" for item in exclude)
        archive_command_lines.extend(
            [
                f"if [ -e {shlex.quote(source)} ]; then",
                f"  mkdir -p {mkdir_target}",
                f"  if [ -d {shlex.quote(source)} ]; then",
                f"    mkdir -p {target}",
                f"    tar -C {shlex.quote(source)} {exclude_args} -cf - . | "
                f"tar -C {target} -xf -",
                "  else",
                f"    cp {shlex.quote(source)} {target}",
                "  fi",
                "  added=1",
                "fi",
            ]
        )
    archive_command_lines.extend(
        [
            'if [ "$added" -eq 0 ]; then exit 42; fi',
            f'tar -czf {shlex.quote(remote_tar)} -C "$tmp" .',
        ]
    )
    archive_command = "\n".join(archive_command_lines)
    try:
        result = await agent_client.execute_command(
            sandbox_id=agent_sandbox_id,
            command=archive_command,
        )
        if result.exit_code == 42:
            return
        if result.exit_code:
            raise RuntimeError(result.stderr or result.stdout or "tar failed")
        await agent_client.download_file(agent_sandbox_id, remote_tar, str(local_tar))
        await upload_harbor_archive(
            verifier_client, verifier_sandbox_id, local_tar, remote_tar
        )
    finally:
        local_tar.unlink(missing_ok=True)
        await agent_client.execute_command(
            sandbox_id=agent_sandbox_id,
            command=f"rm -f {shlex.quote(remote_tar)}",
        )


async def upload_harbor_tests(
    client: SandboxClient, sandbox_id: str, task_dir: Path
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)
    try:
        await build_harbor_tests_archive(task_dir, tar_path)
        remote_tar = "/tmp/harbor_tests.tar.gz"
        await upload_harbor_archive(
            client,
            sandbox_id,
            tar_path,
            remote_tar,
            before_extract="mkdir -p /oracle /tests /logs/verifier",
            timeout=900,
        )
    finally:
        tar_path.unlink(missing_ok=True)


async def upload_harbor_archive(
    client: SandboxClient,
    sandbox_id: str,
    local_tar: Path,
    remote_tar: str,
    *,
    before_extract: str | None = None,
    timeout: int | None = None,
) -> None:
    await client.upload_file(sandbox_id, remote_tar, str(local_tar))
    extract = f"tar -xzf {shlex.quote(remote_tar)} -C / && rm {shlex.quote(remote_tar)}"
    command = f"{before_extract} && {extract}" if before_extract else extract
    result = await client.execute_command(
        sandbox_id=sandbox_id,
        command=command,
        timeout=timeout,
    )
    if result.exit_code:
        raise RuntimeError(result.stderr or result.stdout or "tar extract failed")


async def build_harbor_tests_archive(task_dir: Path, tar_path: Path) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        for dirname, arc_root in (("solution", "oracle"), ("tests", "tests")):
            root = task_dir / dirname
            if not root.exists():
                continue
            for item in root.iterdir():
                tar.add(item, arcname=f"{arc_root}/{item.name}")


def parse_reward_text(reward_text: str) -> float:
    if not reward_text:
        return 0.0
    try:
        return float(reward_text)
    except ValueError:
        pass
    try:
        data = json.loads(reward_text)
    except json.JSONDecodeError:
        return 0.0
    if not isinstance(data, Mapping):
        return 0.0
    return float(data.get("reward", 0.0))
