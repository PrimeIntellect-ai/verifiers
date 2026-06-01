import os
import re
import shlex
import tempfile
from pathlib import Path, PurePosixPath
from typing import cast

import verifiers as vf
from verifiers.utils.import_utils import load_toml
from verifiers.v1.utils.sandbox_utils import SandboxClient, create_sandbox_lease

from tasksets.utils.harbor_utils import (
    TASKS_SUBDIR,
    bundle_tasks_root,
    download_harbor_dataset,
    harbor_sandbox,
    harbor_task_dirs,
    parse_gb,
    parse_number,
    parse_reward_text,
    upload_harbor_tests,
)

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

HARBOR_DEFAULT_SANDBOX = vf.SandboxConfig(
    image="python:3.11-slim",
    cpu_cores=2.0,
    memory_gb=4.0,
    disk_size_gb=10.0,
    timeout_minutes=120,
    workdir="/app",
    command_timeout=900,
)


class HarborTasksetConfig(vf.TasksetConfig):
    taskset_id: str | None = "harbor"
    dataset: str | None = None
    bundle_package: str | None = None
    task_names: list[str] | None = None
    cache_dir: str | None = None
    refresh: bool = False
    sandbox: vf.SandboxConfig = HARBOR_DEFAULT_SANDBOX
    verifier_timeout_seconds: float = 900.0
    task_dir: str = "/task"
    env: dict[str, str] = {}


class HarborTaskset(vf.Taskset[HarborTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        config = self.config
        assert split in ("train", "eval")
        if config.dataset is not None:
            cache_dir_path = (
                Path(str(config.cache_dir)).expanduser() if config.cache_dir else None
            )
            root = download_harbor_dataset(
                config.dataset,
                cache_dir=cache_dir_path,
                refresh=config.refresh,
            )
        else:
            bundle_package = config.bundle_package
            if bundle_package is None:
                raise RuntimeError(
                    "HarborTaskset() without a dataset requires bundle_package. "
                    "Pass dataset='...' to fetch from Harbor Hub, or set "
                    "bundle_package=__name__ from the package that owns tasks/."
                )
            root = bundle_tasks_root(bundle_package)
            if not root.exists():
                raise FileNotFoundError(
                    "HarborTaskset() without a dataset requires "
                    f"{bundle_package}/{TASKS_SUBDIR}/ to contain Harbor task "
                    f"directories. Not found: {root}"
                )
        rows = [
            harbor_task_row(config, task_dir, index)
            for index, task_dir in enumerate(
                harbor_task_dirs(root, list(config.task_names or []))
            )
        ]
        assert rows, f"No valid Harbor tasks found in {root}."
        return rows

    @vf.reward(weight=1.0)
    async def harbor_reward(self, task: vf.Task, state: vf.State) -> float:
        return await harbor_reward(task, state)


def harbor_task_row(
    config: HarborTasksetConfig, task_dir: Path, index: int
) -> vf.ConfigData:
    task_toml_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    with task_toml_path.open("rb") as f:
        task_config = load_toml(f)
    environment = task_config.get("environment", {}) or {}
    if not isinstance(environment, dict):
        raise TypeError(f"{task_toml_path} [environment] must be a mapping.")
    agent_config = task_config.get("agent", {}) or {}
    verifier_config = task_config.get("verifier", {}) or {}
    if not isinstance(agent_config, dict):
        raise TypeError(f"{task_toml_path} [agent] must be a mapping.")
    if not isinstance(verifier_config, dict):
        raise TypeError(f"{task_toml_path} [verifier] must be a mapping.")

    test_timeout = parse_number(
        verifier_config.get("timeout_sec"), config.verifier_timeout_seconds
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
        and not isinstance(verifier_environment, dict)
    ):
        raise TypeError(f"{task_toml_path} [verifier.environment] must be a mapping.")

    sandbox = harbor_sandbox_config(
        config,
        cast(vf.ConfigData, dict(environment)),
        int(parse_number(agent_config.get("timeout_sec"), 900)),
    )
    verifier_sandbox: vf.ConfigData | None = None
    verifier_upload_tests = False
    if verifier_mode == VERIFIER_MODE_SEPARATE and verifier_environment is None:
        verifier_sandbox = {**sandbox, "command_timeout": int(test_timeout)}
        verifier_upload_tests = True
    elif verifier_mode == VERIFIER_MODE_SEPARATE:
        verifier_sandbox = harbor_sandbox_config(
            config,
            cast(vf.ConfigData, dict(cast(dict[str, object], verifier_environment))),
            int(test_timeout),
        )

    instruction = instruction_path.read_text().strip()
    task_remote_dir = config.task_dir.rstrip("/") or "/task"
    workdir = sandbox.get("workdir") or "/app"
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
                "AGENT_WORKDIR": str(workdir),
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
    config: HarborTasksetConfig,
    environment: vf.ConfigData,
    command_timeout: int,
) -> vf.ConfigData:
    sandbox = harbor_sandbox(HARBOR_DEFAULT_SANDBOX, config.sandbox)
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
    updates: dict[str, object] = {
        "image": environment.get("docker_image") or sandbox.image,
        "cpu_cores": parse_number(environment.get("cpus"), sandbox.cpu_cores),
        "memory_gb": parse_gb(memory, sandbox.memory_gb),
        "disk_size_gb": parse_gb(storage, sandbox.disk_size_gb),
        "command_timeout": command_timeout,
        "scope": sandbox.scope,
    }
    if "allow_internet" in environment:
        updates["network_access"] = bool(environment["allow_internet"])
    return sandbox.model_copy(update=updates).data(fill_defaults=False)


async def harbor_reward(
    task: vf.Task | vf.ConfigData, state: vf.State | vf.ConfigData
) -> float:
    if state.get("error") is not None:
        return 0.0
    sandbox_id = state.get("sandbox_id")
    if not isinstance(sandbox_id, str):
        return 0.0
    harbor = task.get("harbor")
    if not isinstance(harbor, dict):
        return 0.0
    task_dir = Path(str(harbor["task_dir"]))
    mode = str(harbor.get("verifier_mode") or VERIFIER_MODE_SHARED)
    timeout = int(parse_number(harbor.get("test_timeout"), 900))
    verifier_env = harbor_verifier_env(cast(vf.ConfigData, dict(harbor)))
    from prime_sandboxes import AsyncSandboxClient

    client = cast(SandboxClient, AsyncSandboxClient())
    try:
        if mode == VERIFIER_MODE_SEPARATE:
            reward_text = await run_separate_harbor_verifier(
                client,
                sandbox_id,
                cast(vf.ConfigData, dict(harbor)),
                cast(vf.ConfigData, state),
                timeout,
                verifier_env,
            )
        else:
            await upload_harbor_tests(client, sandbox_id, task_dir)
            reward_text = await run_harbor_tests(
                client,
                sandbox_id,
                cast(vf.ConfigData, state),
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


def harbor_verifier_env(harbor: vf.ConfigData) -> dict[str, str] | None:
    raw_env = harbor.get("verifier_env") or {}
    if not isinstance(raw_env, dict):
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
    harbor: vf.ConfigData,
    state: vf.ConfigData,
    timeout: int,
    verifier_env: dict[str, str] | None,
) -> str:
    sandbox = harbor.get("verifier_sandbox")
    if not isinstance(sandbox, dict):
        raise RuntimeError("Separate Harbor verifier did not resolve a sandbox.")
    from prime_sandboxes import AsyncSandboxClient

    verifier_client = cast(SandboxClient, AsyncSandboxClient())
    lease = await create_sandbox_lease(
        vf.SandboxConfig.model_validate(dict(sandbox)), "harbor", client=verifier_client
    )
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
        await verifier_client.aclose()


async def run_harbor_tests(
    client: SandboxClient,
    sandbox_id: str,
    state: vf.ConfigData,
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
    harbor: vf.ConfigData,
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
        elif isinstance(artifact, dict):
            source_value = artifact.get("source")
            if not isinstance(source_value, str):
                raise TypeError("Harbor artifacts must be strings or source mappings.")
            source = source_value
            raw_exclude = artifact.get("exclude") or []
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
    try:
        result = await agent_client.execute_command(
            sandbox_id=agent_sandbox_id,
            command="\n".join(archive_command_lines),
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


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)
