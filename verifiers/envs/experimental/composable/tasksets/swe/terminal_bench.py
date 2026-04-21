from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import yaml

import verifiers as vf
from verifiers.envs.experimental.composable.tasksets.harbor import HarborDatasetTaskSet

DEFAULT_REPO_URL = "https://github.com/harbor-framework/terminal-bench.git"
DEFAULT_GIT_REF = "dataset/terminal-bench-core/v0.2.x"
DEFAULT_TASKS_SUBDIR = "tasks"
DEFAULT_AGENT_WORKDIR = "/app"

_TASK_TOML_TEMPLATE = """\
version = "1.0"

[metadata]
author_name = {author_name}
author_email = {author_email}
difficulty = {difficulty}
category = {category}
tags = {tags}

[verifier]
timeout_sec = {test_timeout}

[agent]
timeout_sec = {agent_timeout}
{harness_toml}

[environment]
docker_image = {docker_image}
start_command = "tail -f /dev/null"
cpus = {cpu_cores}
memory = {memory}
storage = {storage}
gpus = {gpu_count}
"""

_TEST_SH_TEMPLATE = """\
#!/bin/bash
set -u

mkdir -p /logs/verifier /logs/agent
export TEST_DIR=/tests
export T_BENCH_TEST_DIR=/tests
export T_BENCH_CONTAINER_LOGS_PATH=/logs
export T_BENCH_CONTAINER_AGENT_LOGS_PATH=/logs/agent

cd {agent_workdir}
bash /tests/run-tests.sh
status=$?

if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

exit "$status"
"""

_DOCKER_SETUP_PREAMBLE = """\
#!/bin/bash
set -euo pipefail

_tbench_copy() {
  local src="$1"
  local dest="$2"

  if [ -d "$src" ]; then
    mkdir -p "$dest"
    cp -a "$src"/. "$dest"/
    return
  fi

  if [ -e "$src" ]; then
    if [[ "$dest" == */ ]]; then
      mkdir -p "$dest"
      cp -a "$src" "$dest"
    else
      mkdir -p "$(dirname "$dest")"
      cp -a "$src" "$dest"
    fi
    return
  fi

  echo "Missing Dockerfile COPY source: $src" >&2
  exit 1
}

mkdir -p {agent_workdir} /logs/verifier /logs/agent
cd /
"""

_EXCLUDED_SETUP_ASSETS = {
    "Dockerfile",
    "docker-compose.yaml",
    "run-tests.sh",
    "solution.sh",
    "solution.yaml",
    "task.yaml",
    "tests",
}


def _dockerfile_instructions(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []

    instructions: list[tuple[str, str]] = []
    current = ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not current and (not stripped or stripped.startswith("#")):
            continue
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue

        current += line
        stripped_current = current.strip()
        current = ""
        if not stripped_current or stripped_current.startswith("#"):
            continue

        command, _, rest = stripped_current.partition(" ")
        instructions.append((command.upper(), rest.strip()))
    return instructions


class TerminalBench2TaskSet(HarborDatasetTaskSet):
    """Terminal-Bench 2 tasks adapted to the Harbor taskset contract."""

    default_workdir = DEFAULT_AGENT_WORKDIR

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        repo_url: str = DEFAULT_REPO_URL,
        git_ref: str = DEFAULT_GIT_REF,
        tasks_subdir: str = DEFAULT_TASKS_SUBDIR,
        task_ids: list[str] | None = None,
        tasks: list[str] | None = None,
        max_examples: int = -1,
        agent_workdir: str = DEFAULT_AGENT_WORKDIR,
        cpu_cores: int = 2,
        memory_gb: int = 4,
        disk_size_gb: int = 10,
        gpu_count: int = 0,
        harness_config: dict[str, Any] | None = None,
        docker_image_prefix: str | None = None,
        docker_image_tag: str = "latest",
        replay_dockerfile: bool = True,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        name: str = "terminal-bench/terminal-bench-2",
    ):
        selected_tasks = task_ids or tasks
        self.source_dataset_path = self.resolve_dataset_path(
            dataset_path=dataset_path,
            repo_url=repo_url,
            git_ref=git_ref,
            tasks_subdir=tasks_subdir,
            cache_dir=cache_dir,
            force_download=force_download,
        )
        self.task_ids = set(selected_tasks or [])
        self.max_examples = max_examples
        self.agent_workdir = agent_workdir
        self.cpu_cores = cpu_cores
        self.memory_gb = memory_gb
        self.disk_size_gb = disk_size_gb
        self.gpu_count = gpu_count
        self.docker_image_prefix = docker_image_prefix
        self.docker_image_tag = docker_image_tag
        self.replay_dockerfile = replay_dockerfile
        self.harness_config = dict(
            harness_config
            or {
                "agent": "openclaw",
                "agent_workdir": agent_workdir,
            }
        )
        self._generated_tasks_dir = tempfile.TemporaryDirectory(
            prefix="terminal_bench_2_harbor_"
        )
        self.generated_tasks_path = Path(self._generated_tasks_dir.name)
        self.generate_harbor_tasks()
        super().__init__(
            dataset_path=self.generated_tasks_path,
            agent_workdir=agent_workdir,
            name=name,
        )

    def resolve_dataset_path(
        self,
        *,
        dataset_path: str | Path | None,
        repo_url: str,
        git_ref: str,
        tasks_subdir: str,
        cache_dir: str | Path | None,
        force_download: bool,
    ) -> Path:
        if dataset_path is not None:
            return Path(dataset_path).expanduser()

        root = Path(cache_dir or Path.home() / ".cache" / "verifiers")
        safe_ref = re.sub(r"[^A-Za-z0-9_.-]+", "-", git_ref).strip("-") or "head"
        target = root / "terminal-bench-2" / safe_ref
        if force_download and target.exists():
            shutil.rmtree(target)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="terminal_bench_2_clone_") as tmp:
                clone_dir = Path(tmp) / "repo"
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth=1",
                        "--branch",
                        git_ref,
                        repo_url,
                        str(clone_dir),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                shutil.copytree(clone_dir / tasks_subdir, target)
        return target

    def generate_harbor_tasks(self) -> None:
        generated_count = 0
        for source_task_dir in sorted(self.source_dataset_path.iterdir()):
            if not source_task_dir.is_dir():
                continue
            if self.task_ids and source_task_dir.name not in self.task_ids:
                continue

            self.generate_harbor_task(source_task_dir)
            generated_count += 1
            if self.max_examples > -1 and generated_count >= self.max_examples:
                break

        if generated_count == 0:
            raise ValueError("No Terminal-Bench tasks matched the requested filters.")

    def generate_harbor_task(self, source_task_dir: Path) -> Path:
        config = yaml.safe_load((source_task_dir / "task.yaml").read_text())

        task_dir = self.generated_tasks_path / source_task_dir.name
        (task_dir / "environment").mkdir(parents=True, exist_ok=True)
        (task_dir / "tests").mkdir(parents=True, exist_ok=True)
        (task_dir / "solution").mkdir(parents=True, exist_ok=True)

        docker_image = self.build_docker_image(source_task_dir)
        (task_dir / "instruction.md").write_text(str(config["instruction"]).strip())
        (task_dir / "task.toml").write_text(self.render_task_toml(config, docker_image))
        (task_dir / "info.json").write_text(
            json.dumps(
                {
                    "source_task_dir": str(source_task_dir),
                    "task_yaml": config,
                    "docker_image": docker_image,
                    "dockerfile_setup_script": self.render_dockerfile_setup_script(
                        source_task_dir / "Dockerfile"
                    ),
                },
                indent=2,
                default=str,
            )
        )
        if (source_task_dir / "Dockerfile").exists():
            shutil.copy2(source_task_dir / "Dockerfile", task_dir / "environment")

        self.write_tests(source_task_dir, task_dir / "tests")
        self.write_solution(source_task_dir, task_dir / "solution" / "solve.sh")
        return task_dir

    def build_docker_image(self, source_task_dir: Path) -> str:
        if self.docker_image_prefix:
            return (
                f"{self.docker_image_prefix.rstrip('/')}/"
                f"{source_task_dir.name}:{self.docker_image_tag}"
            )
        for instruction, rest in reversed(
            _dockerfile_instructions(source_task_dir / "Dockerfile")
        ):
            if instruction == "FROM":
                for token in shlex.split(rest):
                    lowered = token.lower()
                    if token.startswith("--"):
                        continue
                    if lowered == "as":
                        break
                    return token
                raise ValueError(f"Could not parse Dockerfile FROM line: {rest}")
        return "ghcr.io/laude-institute/t-bench/python-3-13:20250620"

    def render_task_toml(self, config: dict[str, Any], docker_image: str) -> str:
        return _TASK_TOML_TEMPLATE.format(
            author_name=self.toml_value(config.get("author_name", "unknown")),
            author_email=self.toml_value(config.get("author_email", "unknown")),
            difficulty=self.toml_value(config.get("difficulty", "unknown")),
            category=self.toml_value(config.get("category", "terminal")),
            tags=self.toml_value(config.get("tags", [])),
            test_timeout=float(config.get("max_test_timeout_sec", 60.0)),
            agent_timeout=float(config.get("max_agent_timeout_sec", 360.0)),
            harness_toml=self.render_harness_toml(),
            docker_image=self.toml_value(docker_image),
            cpu_cores=self.cpu_cores,
            memory=self.toml_value(f"{self.memory_gb}G"),
            storage=self.toml_value(f"{self.disk_size_gb}G"),
            gpu_count=self.gpu_count,
        )

    def render_harness_toml(self) -> str:
        lines = ["[agent.harness]"]
        for key, value in self.harness_config.items():
            if value is not None:
                lines.append(f"{key} = {self.toml_value(value)}")
        return "\n".join(lines)

    def toml_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int | float):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(self.toml_value(item) for item in value) + "]"
        return json.dumps(str(value))

    def write_tests(self, source_task_dir: Path, tests_dir: Path) -> None:
        source_tests = source_task_dir / "tests"
        if source_tests.exists():
            for item in source_tests.iterdir():
                destination = tests_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, destination)
                else:
                    shutil.copy2(item, destination)
        shutil.copy2(source_task_dir / "run-tests.sh", tests_dir / "run-tests.sh")
        test_sh = tests_dir / "test.sh"
        test_sh.write_text(_TEST_SH_TEMPLATE.format(agent_workdir=self.agent_workdir))
        test_sh.chmod(0o755)

    def write_solution(self, source_task_dir: Path, solve_sh: Path) -> None:
        solution_sh = source_task_dir / "solution.sh"
        solution_yaml = source_task_dir / "solution.yaml"
        if solution_sh.exists():
            shutil.copy2(solution_sh, solve_sh)
        elif solution_yaml.exists():
            commands = yaml.safe_load(solution_yaml.read_text()) or []
            lines = ["#!/bin/bash", "set -euo pipefail", ""]
            for item in commands:
                command = item["command"] if isinstance(item, dict) else item
                lines.append(str(command))
            lines.append("")
            solve_sh.write_text("\n".join(lines))
        else:
            solve_sh.write_text("#!/bin/bash\nset -euo pipefail\n")
        solve_sh.chmod(0o755)

    def render_dockerfile_setup_script(self, dockerfile: Path) -> str:
        lines = [
            _DOCKER_SETUP_PREAMBLE.replace(
                "{agent_workdir}", shlex.quote(self.agent_workdir)
            )
        ]
        workdir = "/"
        for instruction, rest in _dockerfile_instructions(dockerfile):
            if instruction == "FROM":
                continue
            if instruction == "WORKDIR":
                workdir = self.resolve_container_path(rest, workdir)
                lines.append(f"mkdir -p {shlex.quote(workdir)}")
                lines.append(f"cd {shlex.quote(workdir)}")
            elif instruction == "RUN":
                lines.append(rest)
            elif instruction in {"COPY", "ADD"}:
                lines.extend(self.render_copy_commands(rest, workdir))
            elif instruction == "ENV":
                tokens = shlex.split(rest)
                if tokens and all("=" in token for token in tokens):
                    lines.extend(f"export {token}" for token in tokens)
                elif len(tokens) >= 2:
                    lines.append(
                        f"export {tokens[0]}={shlex.quote(' '.join(tokens[1:]))}"
                    )
            elif instruction == "ARG":
                token = rest.strip()
                if "=" in token:
                    lines.append(f"export {token}")
            elif instruction in {"CMD", "ENTRYPOINT", "USER", "EXPOSE", "VOLUME"}:
                continue
            else:
                lines.append(f"# Unsupported Dockerfile instruction: {instruction}")
        lines.append(f"mkdir -p {shlex.quote(self.agent_workdir)}")
        lines.append("")
        return "\n".join(lines)

    def render_copy_commands(self, rest: str, workdir: str) -> list[str]:
        tokens = (
            [str(item) for item in json.loads(rest)]
            if rest.startswith("[")
            else shlex.split(rest)
        )
        while tokens and tokens[0].startswith("--"):
            tokens.pop(0)
        if len(tokens) < 2:
            return []

        dest = self.resolve_container_path(tokens[-1], workdir)
        lines: list[str] = []
        for source in tokens[:-1]:
            source_path = source[2:] if source.startswith("./") else source
            host_source = f"/tmp/tbench_task_src/{source_path}"
            if any(char in host_source for char in "*?["):
                lines.append(
                    f"for _src in {host_source}; do "
                    f'_tbench_copy "$_src" {shlex.quote(dest)}; done'
                )
            else:
                lines.append(
                    f"_tbench_copy {shlex.quote(host_source)} {shlex.quote(dest)}"
                )
        return lines

    def resolve_container_path(self, path: str, workdir: str) -> str:
        stripped = path.strip()
        if stripped.startswith("/"):
            return stripped
        return f"{workdir.rstrip('/')}/{stripped}".replace("//", "/")

    async def setup(self, state: vf.State) -> None:
        await super().setup(state)
        info = state["info"]
        if not self.replay_dockerfile:
            return

        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await self.upload_setup_assets(
            sandbox_client=sandbox_client,
            sandbox_id=sandbox_id,
            source_task_dir=Path(info["source_task_dir"]),
        )
        setup_script = info.get("dockerfile_setup_script") or ""
        if setup_script.strip():
            remote_script = "/tmp/tbench_dockerfile_setup.sh"
            with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as file:
                script_path = Path(file.name)
                file.write(setup_script)
            try:
                await sandbox_client.upload_file(
                    sandbox_id, remote_script, str(script_path)
                )
                result = await sandbox_client.execute_command(
                    sandbox_id,
                    f"bash {remote_script} && rm -rf /tmp/tbench_task_src {remote_script}",
                    working_dir=None,
                    timeout=max(900, int(info["config"]["agent"]["timeout_sec"])),
                )
                if result.exit_code != 0:
                    output = (result.stdout or "") + (result.stderr or "")
                    raise vf.SandboxError(
                        f"Terminal-Bench task setup failed "
                        f"(exit={result.exit_code}): {output[:1000]}"
                    )
            finally:
                script_path.unlink(missing_ok=True)

    async def upload_setup_assets(
        self,
        *,
        sandbox_client: Any,
        sandbox_id: str,
        source_task_dir: Path,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)

        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for item in source_task_dir.iterdir():
                    if item.name in _EXCLUDED_SETUP_ASSETS:
                        continue
                    tar.add(item, arcname=item.name)

            remote_tar = "/tmp/tbench_task_src.tar.gz"
            await sandbox_client.upload_file(sandbox_id, remote_tar, str(tar_path))
            await sandbox_client.execute_command(
                sandbox_id,
                "rm -rf /tmp/tbench_task_src && mkdir -p /tmp/tbench_task_src && "
                f"tar -xzf {remote_tar} -C /tmp/tbench_task_src && rm {remote_tar}",
                working_dir=None,
                timeout=900,
            )
        finally:
            tar_path.unlink(missing_ok=True)


TerminalBenchTaskSet = TerminalBench2TaskSet
