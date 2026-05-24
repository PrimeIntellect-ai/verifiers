import base64
import hashlib
import io
import json
import keyword
import os
import random
import re
import shlex
import tarfile
import textwrap
from collections.abc import Callable, Mapping
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

from verifiers.envs.experimental.utils.git_checkout_cache import (
    resolve_git_checkout,
    validate_git_checkout,
)

from ...config import SandboxConfig, sandbox_config_mapping
from ...harness import Harness
from ...state import State
from ...task import Task
from ...taskset import Taskset
from ...utils.prompt_utils import task_text
from .command import command_program
from .configs import (
    RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
    RLMConfig,
)
from ...types import ConfigData, ConfigMap, ProgramCommand, ProgramValue

DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_SKILLS_PATH = "/task/rlm-skills"
DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH = "/tmp/vf-rlm-tool-skills.tar.gz.b64"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
REQUIRED_RLM_CHECKOUT_FILES = ("install.sh", "pyproject.toml")
ProgramDir = str | Path | Traversable


class RLM(Harness):
    def __init__(self, config: RLMConfig | None = None):
        harness_config = RLMConfig() if config is None else config
        assert isinstance(harness_config, RLMConfig)
        super().__init__(config=harness_config.model_copy(update={"program": None}))
        self.config = harness_config
        if (
            not harness_config.include_sub_rlm_trajectories
            and harness_config.keep_trajectory_step is None
        ):
            self.keep_trajectory_step = keep_only_parent_rlm_steps
        summarize_resolver = build_summarize_resolver(
            harness_config.summarize_at_tokens
        )
        env: dict[str, ProgramValue] = {
            "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "OPENAI_MODEL": "runtime.model",
            "RLM_MODEL": "runtime.model",
            "RLM_TOOLS": ",".join(harness_config.rlm_tools),
            "RLM_MAX_TURNS": str(harness_config.rlm_max_turns),
            "RLM_EXEC_TIMEOUT": str(harness_config.rlm_exec_timeout),
            "RLM_MAX_DEPTH": str(harness_config.rlm_max_depth),
            **harness_config.env_vars,
        }
        if summarize_resolver is not None:
            env["RLM_SUMMARIZE_AT_TOKENS"] = summarize_resolver
        sandbox_config: ConfigMap | SandboxConfig | bool
        setup_timeout = max(harness_config.rlm_exec_timeout + 120, 600)
        sandbox_config = harness_config.sandbox or True
        if sandbox_config is True:
            sandbox_config = {
                "image": "python:3.11-slim",
                "workdir": harness_config.workdir,
                "cpu_cores": 1,
                "memory_gb": 2,
                "disk_size_gb": 5,
                "network_access": True,
                "timeout_minutes": 60,
                "command_timeout": max(harness_config.rlm_exec_timeout + 120, 600),
                "setup_timeout": setup_timeout,
            }
        elif sandbox_config is not False:
            sandbox_options = sandbox_config_mapping(sandbox_config) or {}
            if isinstance(sandbox_options.get("setup_timeout"), int):
                setup_timeout = cast(int, sandbox_options["setup_timeout"])
            sandbox_config = {
                "workdir": harness_config.workdir,
                "command_timeout": max(harness_config.rlm_exec_timeout + 120, 600),
                "setup_timeout": setup_timeout,
                **sandbox_options,
            }
        dirs: dict[str, ProgramValue] = {
            DEFAULT_RLM_CHECKOUT_PATH: rlm_checkout_loader(
                local_checkout=harness_config.local_checkout,
                rlm_repo_url=harness_config.rlm_repo_url,
                rlm_repo_ref=harness_config.rlm_repo_ref,
                gh_token=harness_config.gh_token,
            )
        }
        self._skills_dir: ProgramDir | None = (
            Path(harness_config.skills) if harness_config.skills is not None else None
        )
        if self._skills_dir is not None:
            dirs[DEFAULT_RLM_SKILLS_PATH] = self._skills_dir
        command: ProgramCommand = [
            "bash",
            "-lc",
            build_run_script(harness_config.instruction_path, harness_config.workdir),
        ]
        program = command_program(
            command=command,
            sandbox=sandbox_config,
            files={
                harness_config.instruction_path: task_instruction_text,
                RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH: (
                    harness_config.append_to_system_prompt
                ),
                DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH: self.vf_tool_skills_archive,
            },
            dirs=dirs,
            setup=[
                "apt-get -o Acquire::Retries=3 update && "
                "apt-get -o Acquire::Retries=3 install -y --no-install-recommends "
                "ca-certificates curl git && rm -rf /var/lib/apt/lists/*",
                (
                    f"if [ -s {shlex.quote(DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH)} ]; then "
                    f"mkdir -p {shlex.quote(DEFAULT_RLM_SKILLS_PATH)} && "
                    f"base64 -d {shlex.quote(DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH)} | "
                    f"tar -xzf - -C {shlex.quote(DEFAULT_RLM_SKILLS_PATH)}; "
                    "fi"
                ),
                "bash -lc "
                + shlex.quote(
                    f"""
set -eo pipefail
export RLM_CHECKOUT_PATH={shlex.quote(DEFAULT_RLM_CHECKOUT_PATH)}
test -f "$RLM_CHECKOUT_PATH/install.sh"
bash "$RLM_CHECKOUT_PATH/install.sh"
"""
                ),
            ],
            env=env,
            artifacts={
                "rlm_metrics": {
                    "path": f"{harness_config.workdir}/.rlm/sessions/*/meta.json",
                    "format": "json",
                    "key": "metrics",
                    "optional": True,
                }
            },
            program=harness_config.program,
        )
        program["setup_timeout"] = setup_timeout
        self._configure_runtime(
            program=program,
            sandbox=None if sandbox_config is False else sandbox_config,
            metrics=[
                rlm_sub_llm_call_count,
                rlm_sub_llm_total_turns,
                rlm_sub_llm_total_tool_calls,
            ],
        )

    def attach_taskset(self, taskset: Taskset) -> None:
        if self._skills_dir is None:
            upload_dirs = taskset.get_upload_dirs()
            if not isinstance(upload_dirs, Mapping):
                raise TypeError("Taskset.get_upload_dirs() must return a mapping.")
            self._skills_dir = cast(ProgramDir | None, upload_dirs.get("skills"))
            if self._skills_dir is not None:
                if not isinstance(self.program, Mapping):
                    raise TypeError("RLM program must be a mapping.")
                program = dict(cast(ConfigMap, self.program))
                dirs = dict(cast(ConfigMap, program.get("dirs") or {}))
                dirs[DEFAULT_RLM_SKILLS_PATH] = self._skills_dir
                program["dirs"] = dirs
                self.program = program
        super().attach_taskset(taskset)
        self._program = self.compile_program(self.program)

    def vf_tool_skills_archive(self, state: State) -> str:
        tool_defs = self.runtime.tool_defs(state) or []
        if not tool_defs:
            return ""
        used_names: set[str] = set()
        if self._skills_dir is not None:
            root = (
                Path(self._skills_dir)
                if isinstance(self._skills_dir, str)
                else self._skills_dir
            )
            used_names.update(child.name for child in root.iterdir() if child.is_dir())
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for raw_tool_def in tool_defs:
                tool_def = cast(ConfigData, raw_tool_def.model_dump())
                tool_name = str(tool_def["name"])
                skill_name = re.sub(r"\W", "_", tool_name)
                if not skill_name or skill_name[0].isdigit():
                    skill_name = f"tool_{skill_name}"
                if keyword.iskeyword(skill_name):
                    skill_name = f"{skill_name}_tool"
                base_name = skill_name
                index = 2
                while skill_name in used_names:
                    skill_name = f"{base_name}_{index}"
                    index += 1
                used_names.add(skill_name)
                description = str(
                    tool_def.get("description")
                    or f"Call the {tool_name} verifier tool."
                )
                schema = json.dumps(
                    tool_def.get("parameters") or {}, indent=2, sort_keys=True
                )
                parameters = cast(ConfigData, tool_def.get("parameters") or {})
                properties = cast(ConfigData, parameters.get("properties") or {})
                allowed_arguments = (
                    sorted(properties)
                    if parameters.get("additionalProperties") is False
                    else None
                )
                signature, arguments = render_run_signature_and_arguments(tool_def)
                module = textwrap.dedent(
                    f"""\
                    import os
                    import requests


                    TOOL_ALLOWED_ARGUMENTS = {allowed_arguments!r}


                    async def run({signature}):
                        {json.dumps(description)}
                        arguments = _tool_arguments({arguments}, kwargs)
                        base = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                        if not base:
                            raise RuntimeError("No Verifiers endpoint URL is configured.")
                        api_key = (
                            os.environ.get("OPENAI_API_KEY")
                            or os.environ.get("ANTHROPIC_API_KEY")
                            or "intercepted"
                        )
                        response = requests.post(
                            f"{{base.rsplit('/v1', 1)[0].rstrip('/')}}/vf/tools/" + {tool_name!r},
                            json={{"arguments": arguments}},
                            headers={{"Authorization": f"Bearer {{api_key}}"}},
                            timeout=300,
                        )
                        payload = _response_payload(response)
                        return payload.get("result")


                    def _tool_arguments(arguments: dict | None, kwargs: dict) -> dict:
                        merged = {{**(arguments or {{}}), **kwargs}}
                        if TOOL_ALLOWED_ARGUMENTS is None:
                            return merged
                        return {{key: merged[key] for key in TOOL_ALLOWED_ARGUMENTS if key in merged}}


                    def _response_payload(response):
                        if not response.content:
                            response.raise_for_status()
                            return {{}}
                        payload = response.json()
                        if "error" in payload:
                            raise RuntimeError(str(payload["error"]))
                        response.raise_for_status()
                        return payload
                    """
                )
                distribution_name = skill_name.replace("_", "-")
                files = {
                    f"{skill_name}/SKILL.md": f"""# {skill_name}

{description}

This skill calls the Verifiers V1 tool `{tool_name}` for the current rollout.

Pass tool arguments as a dictionary:

```python
result = await {skill_name}({{"argument_name": "value"}})
```

Tool schema:

```json
{schema}
```
""",
                    f"{skill_name}/pyproject.toml": textwrap.dedent(
                        f"""\
                        [project]
                        name = "rlm-skill-{distribution_name}"
                        version = "0.0.0"
                        dependencies = ["requests", "rlm"]

                        [project.scripts]
                        {skill_name} = "rlm.skill:cli"

                        [build-system]
                        requires = ["hatchling"]
                        build-backend = "hatchling.build"

                        [tool.hatch.build.targets.wheel]
                        packages = ["src/{skill_name}"]
                        """
                    ),
                    f"{skill_name}/src/{skill_name}/__init__.py": (
                        f"from .{skill_name} import run\n\n__all__ = ['run']\n"
                    ),
                    f"{skill_name}/src/{skill_name}/{skill_name}.py": module,
                }
                for path, content in files.items():
                    data = content.encode()
                    info = tarfile.TarInfo(path)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        return base64.b64encode(buffer.getvalue()).decode()


def render_run_signature_and_arguments(tool_def: ConfigMap) -> tuple[str, str]:
    parameters = cast(ConfigData, tool_def.get("parameters") or {})
    properties = cast(ConfigData, parameters.get("properties") or {})
    required = set(cast(list[str], parameters.get("required") or []))
    if any(not valid_tool_parameter(name) for name in properties):
        return "arguments: dict | None = None, **kwargs", "arguments"
    items = sorted(
        properties.items(),
        key=lambda item: (
            "default" in cast(ConfigData, item[1]) or item[0] not in required
        ),
    )
    signature_parts: list[str] = []
    argument_parts: list[str] = []
    for name, raw_schema in items:
        schema = cast(ConfigData, raw_schema)
        if "default" in schema:
            signature_parts.append(f"{name}={schema['default']!r}")
        elif name in required:
            signature_parts.append(name)
        else:
            signature_parts.append(f"{name}=None")
        argument_parts.append(f"{name!r}: {name}")
    signature_parts.append("**kwargs")
    return ", ".join(signature_parts), "{" + ", ".join(argument_parts) + "}"


def valid_tool_parameter(name: str) -> bool:
    return (
        name.isidentifier()
        and not name.startswith("_")
        and not keyword.iskeyword(name)
        and name
        not in {
            "arguments",
            "kwargs",
        }
    )


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)


def build_run_script(instruction_path: str, workdir: str) -> str:
    return f"""
set -eo pipefail
export PATH="$HOME/.local/bin:${{AGENT_PATH:-$PATH}}"
export RLM_MODEL="${{RLM_MODEL:-$OPENAI_MODEL}}"
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH)} 2>/dev/null || true)"
cd "${{AGENT_WORKDIR:-{workdir}}}"
rlm "$(cat {shlex.quote(instruction_path)})"
"""


def rlm_checkout_loader(
    local_checkout: str | Path | None,
    rlm_repo_url: str,
    rlm_repo_ref: str,
    gh_token: str | None,
) -> Callable[[], Path]:
    checkout: Path | None = None

    def load() -> Path:
        nonlocal checkout
        if checkout is not None:
            return checkout
        if local_checkout is not None:
            checkout = validate_git_checkout(
                Path(local_checkout),
                required_files=REQUIRED_RLM_CHECKOUT_FILES,
            )
        else:
            checkout = resolve_git_checkout(
                repo_url=rlm_repo_url,
                ref=rlm_repo_ref,
                cache_root=DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT,
                gh_token=gh_token or os.environ.get("GH_TOKEN"),
                required_files=REQUIRED_RLM_CHECKOUT_FILES,
            )
        return checkout

    return load


def task_instruction_text(task: Task, state: State) -> str:
    return task_text(task, state, keys=("instruction", "question"))


def keep_only_parent_rlm_steps(step: object, state: State, headers: ConfigMap) -> bool:
    _ = step, state
    return str(headers.get("x-rlm-depth", "0")) == "0"


def rlm_metric(state: ConfigMap, key: str) -> float:
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return 0.0
    artifacts = cast(ConfigMap, artifacts)
    metrics = artifacts.get("rlm_metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    metrics = cast(ConfigMap, metrics)
    value = metrics.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return 0.0
    return float(value or 0.0)


async def rlm_sub_llm_call_count(task: Task, state: State) -> float:
    _ = task
    return rlm_metric(state, "sub_llm_call_count")


async def rlm_sub_llm_total_turns(task: Task, state: State) -> float:
    _ = task
    return rlm_metric(state, "sub_llm_total_turns")


async def rlm_sub_llm_total_tool_calls(task: Task, state: State) -> float:
    _ = task
    return rlm_metric(state, "sub_llm_total_tool_calls")


def build_summarize_resolver(
    value: int | tuple[int, int] | list[int] | None,
) -> Callable[..., str | None] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("summarize_at_tokens must be an int or (lo, hi) pair")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("summarize_at_tokens must be positive")

        def fixed_threshold(state: State) -> str:
            _ = state
            return str(value)

        return fixed_threshold
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("summarize_at_tokens pair must have 2 elements")
        lo, hi = int(value[0]), int(value[1])
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError("summarize_at_tokens pair must satisfy 0 < lo <= hi")

        def sampled_threshold(state: State) -> str:
            prompt = json.dumps(state.get("prompt"), sort_keys=True, default=str)
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            return str(random.Random(int(digest[:16], 16)).randint(lo, hi))

        return sampled_threshold
    raise ValueError("summarize_at_tokens must be int, (lo, hi), or None")
