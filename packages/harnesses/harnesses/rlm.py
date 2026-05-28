import base64
import io
import inspect
import json
import keyword
import os
import re
import shlex
import tarfile
import textwrap
from collections.abc import Callable, Mapping
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

from pydantic import field_validator

from verifiers.envs.experimental.utils.git_checkout_cache import (
    resolve_git_checkout,
    validate_git_checkout,
)
from verifiers.v1.config import CallableEntry, ConfigSource
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.program import (
    Program,
    ProgramCommand,
    ProgramOptionMap,
    ProgramSetup,
    ProgramValue,
)
from verifiers.v1.runtime import Runtime
from verifiers.v1.sandbox import SandboxConfig, sandbox_config_mapping
from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset
from verifiers.v1.types import (
    ConfigData,
    ConfigMap,
)
from verifiers.v1.utils.program_utils import int_config
from verifiers.v1.utils.prompt_utils import task_text
from verifiers.v1.utils.config_utils import coerce_config

RLM_DEFAULT_REPO_URL = "github.com/PrimeIntellect-ai/rlm-harness.git"
RLM_DEFAULT_REPO_REF = "main"
RLM_DEFAULT_MAX_TURNS = 100
RLM_DEFAULT_EXEC_TIMEOUT = 300
RLM_DEFAULT_MAX_DEPTH = 0
RLM_DEFAULT_INSTRUCTION_PATH = "/rlm/instruction.txt"
RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/rlm/append_to_system_prompt.txt"
RLM_DEFAULT_WORKDIR = "/workspace"
RLM_DEFAULT_TOOLS = ["ipython"]
DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_SKILLS_PATH = "/task/rlm-skills"
DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH = "/tmp/vf-rlm-tool-skills.tar.gz.b64"
DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME = ".vf-generated-tool-skills"
DEFAULT_RLM_TOOL_SKILL_MARKER = ".vf-generated-tool-skill"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
REQUIRED_RLM_CHECKOUT_FILES = ("install.sh", "pyproject.toml")


class RLMConfig(HarnessConfig):
    metrics: list[CallableEntry] = [
        "harnesses.rlm:rlm_sub_llm_call_count",
        "harnesses.rlm:rlm_sub_llm_total_turns",
        "harnesses.rlm:rlm_sub_llm_total_tool_calls",
    ]
    keep_trajectory_step: str | None = "harnesses.rlm:keep_only_parent_rlm_steps"
    workdir: str = RLM_DEFAULT_WORKDIR
    instruction_path: str = RLM_DEFAULT_INSTRUCTION_PATH
    rlm_repo_url: str = RLM_DEFAULT_REPO_URL
    rlm_repo_ref: str = RLM_DEFAULT_REPO_REF
    rlm_max_turns: int = RLM_DEFAULT_MAX_TURNS
    rlm_exec_timeout: int = RLM_DEFAULT_EXEC_TIMEOUT
    rlm_max_depth: int = RLM_DEFAULT_MAX_DEPTH
    summarize_at_tokens: int | None = None
    append_to_system_prompt: str = ""
    local_checkout: str | None = None
    gh_token: str | None = None
    rlm_tools: list[str] = RLM_DEFAULT_TOOLS
    env_vars: dict[str, str] = {}
    skills: str | None = None

    @field_validator("summarize_at_tokens")
    @classmethod
    def validate_summarize_at_tokens(cls, value: object) -> object:
        if value is None:
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("summarize_at_tokens must be a positive integer.")
        if value <= 0:
            raise ValueError("summarize_at_tokens must be positive.")
        return value

    @field_validator("env_vars", mode="before")
    @classmethod
    def validate_env_vars(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
        return value


class RLM(Harness[RLMConfig]):
    config: RLMConfig
    taskset_skills_dir: Path | Traversable | None = None

    def __init__(self, config: ConfigSource = None):
        config_value = coerce_config(RLMConfig, config)
        self.command_program_parts = self._program_config(config_value)
        super().__init__(config=config_value)

    def load_program(self) -> Program:
        program, _ = self.command_program_parts
        return program

    def load_sandbox(self) -> ConfigMap | None:
        _, sandbox = self.command_program_parts
        return sandbox

    def skills_dir(self) -> Path | Traversable | None:
        if self.config.skills is not None:
            return Path(self.config.skills)
        return self.taskset_skills_dir

    def attach_taskset(self, taskset: Taskset) -> None:
        if self.config.skills is None:
            upload_dirs = taskset.get_upload_dirs()
            if not isinstance(upload_dirs, Mapping):
                raise TypeError("Taskset.get_upload_dirs() must return a mapping.")
            skills_dir = upload_dirs.get("skills")
            if skills_dir is not None and not isinstance(
                skills_dir, (Path, Traversable)
            ):
                raise TypeError("Taskset upload dir 'skills' must be a path.")
            self.taskset_skills_dir = skills_dir
            if not isinstance(self.program, Mapping):
                raise TypeError("RLM program must be a mapping.")
            program = dict(cast(ConfigMap, self.program))
            dirs = dict(cast(ConfigMap, program.get("dirs") or {}))
            if self.taskset_skills_dir is None:
                dirs.pop(DEFAULT_RLM_SKILLS_PATH, None)
            else:
                dirs[DEFAULT_RLM_SKILLS_PATH] = self.taskset_skills_dir
            if dirs:
                program["dirs"] = dirs
            else:
                program.pop("dirs", None)
            self.program = program
        super().attach_taskset(taskset)
        self._program = self.compile_program(self.program)

    @classmethod
    def _program_config(cls, config: RLMConfig) -> tuple[Program, ConfigData | None]:
        return Harness.command_program_config(
            config,
            command=cls._command(config),
            files=cls._files(config),
            dirs=cls._dirs(config),
            setup=cls._setup(config),
            env=cls._env(config),
            artifacts=cls._artifacts(config),
            sandbox=cls._sandbox(config),
            setup_timeout=cls._setup_timeout(config),
        )

    @classmethod
    def _command(cls, config: RLMConfig) -> ProgramCommand:
        return [
            "bash",
            "-lc",
            cls._run_script(config.instruction_path, config.workdir),
        ]

    @classmethod
    def _files(cls, config: RLMConfig) -> ProgramOptionMap:
        return {
            config.instruction_path: {"fn": "harnesses.rlm:task_instruction_text"},
            RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH: config.append_to_system_prompt,
            DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH: {
                "fn": "harnesses.rlm:rlm_tool_skills_archive"
            },
        }

    @classmethod
    def _dirs(cls, config: RLMConfig) -> ProgramOptionMap:
        dirs: dict[str, ProgramValue] = {
            DEFAULT_RLM_CHECKOUT_PATH: {
                "fn": "harnesses.rlm:rlm_checkout_path",
                **(
                    {"local_checkout": config.local_checkout}
                    if config.local_checkout
                    else {}
                ),
                "rlm_repo_url": config.rlm_repo_url,
                "rlm_repo_ref": config.rlm_repo_ref,
                **({"gh_token": config.gh_token} if config.gh_token else {}),
            }
        }
        if config.skills is not None:
            dirs[DEFAULT_RLM_SKILLS_PATH] = config.skills
        return dirs

    @classmethod
    def _setup(cls, config: RLMConfig) -> ProgramSetup:
        _ = config
        return [
            "apt-get -o Acquire::Retries=3 update && "
            "apt-get -o Acquire::Retries=3 install -y --no-install-recommends "
            "ca-certificates curl git && rm -rf /var/lib/apt/lists/*",
            "bash -lc " + shlex.quote(cls._skills_install_script()),
            "bash -lc " + shlex.quote(cls._checkout_install_script()),
        ]

    @classmethod
    def _env(cls, config: RLMConfig) -> ProgramOptionMap:
        env: dict[str, ProgramValue] = {
            "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "OPENAI_MODEL": "runtime.model",
            "RLM_MODEL": "runtime.model",
            "RLM_TOOLS": ",".join(config.rlm_tools),
            "RLM_MAX_TURNS": str(config.rlm_max_turns),
            "RLM_EXEC_TIMEOUT": str(config.rlm_exec_timeout),
            "RLM_MAX_DEPTH": str(config.rlm_max_depth),
            **config.env_vars,
        }
        if config.summarize_at_tokens is not None:
            env["RLM_SUMMARIZE_AT_TOKENS"] = {
                "fn": "harnesses.rlm:rlm_summarize_at_tokens",
                "value": config.summarize_at_tokens,
            }
        return env

    @classmethod
    def _artifacts(cls, config: RLMConfig) -> ProgramOptionMap:
        return {
            "rlm_metrics": {
                "path": f"{config.workdir}/.rlm/sessions/*/meta.json",
                "format": "json",
                "key": "metrics",
                "optional": True,
            }
        }

    @classmethod
    def _setup_timeout(cls, config: RLMConfig) -> int:
        setup_timeout = max(config.rlm_exec_timeout + 120, 600)
        sandbox_config = config.sandbox or True
        if sandbox_config is not True and sandbox_config is not False:
            explicit_sandbox_options = (
                sandbox_config_mapping(sandbox_config, fill_defaults=False) or {}
            )
            if explicit_sandbox_options.get("setup_timeout") is not None:
                setup_timeout = int_config(
                    explicit_sandbox_options, "setup_timeout", setup_timeout
                )
        return setup_timeout

    @classmethod
    def _sandbox(cls, config: RLMConfig) -> ConfigMap | SandboxConfig | bool:
        setup_timeout = cls._setup_timeout(config)
        sandbox_config: ConfigMap | SandboxConfig | bool = config.sandbox or True
        if sandbox_config is True:
            return {
                "image": "python:3.11-slim",
                "workdir": config.workdir,
                "cpu_cores": 1,
                "memory_gb": 2,
                "disk_size_gb": 5,
                "network_access": True,
                "timeout_minutes": 60,
                "command_timeout": max(config.rlm_exec_timeout + 120, 600),
                "setup_timeout": setup_timeout,
            }
        if sandbox_config is False:
            return False
        sandbox_options = sandbox_config_mapping(sandbox_config) or {}
        return {
            "workdir": config.workdir,
            "command_timeout": max(config.rlm_exec_timeout + 120, 600),
            **sandbox_options,
            "setup_timeout": setup_timeout,
        }

    @classmethod
    def _skills_install_script(cls) -> str:
        return f"""
set -eo pipefail
skills_path={shlex.quote(DEFAULT_RLM_SKILLS_PATH)}
archive_path={shlex.quote(DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH)}
manifest_path="$skills_path/{DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME}"
mkdir -p "$skills_path"
if [ -f "$manifest_path" ]; then
  while IFS= read -r skill_name; do
    case "$skill_name" in ""|.*|*/*|*..*) continue ;; esac
    if [ -f "$skills_path/$skill_name/{DEFAULT_RLM_TOOL_SKILL_MARKER}" ]; then
      rm -rf "$skills_path/$skill_name"
    fi
  done < "$manifest_path"
  rm -f "$manifest_path"
fi
if [ -s "$archive_path" ]; then
  tmp_archive="$(mktemp)"
  trap 'rm -f "$tmp_archive"' EXIT
  base64 -d "$archive_path" > "$tmp_archive"
  tar -tzf "$tmp_archive" \\
    | awk -F/ 'NF > 1 && $1 != "" {{print $1}}' \\
    | sort -u > "$manifest_path"
  tar -xzf "$tmp_archive" -C "$skills_path"
fi
"""

    @classmethod
    def _checkout_install_script(cls) -> str:
        return f"""
set -eo pipefail
export RLM_CHECKOUT_PATH={shlex.quote(DEFAULT_RLM_CHECKOUT_PATH)}
test -f "$RLM_CHECKOUT_PATH/install.sh"
bash "$RLM_CHECKOUT_PATH/install.sh"
"""

    @classmethod
    def _run_script(cls, instruction_path: str, workdir: str) -> str:
        return f"""
set -eo pipefail
export PATH="$HOME/.local/bin:${{AGENT_PATH:-$PATH}}"
export RLM_MODEL="${{RLM_MODEL:-$OPENAI_MODEL}}"
export OPENAI_API_KEY="${{OPENAI_API_KEY:-intercepted}}"
export RLM_APPEND_TO_SYSTEM_PROMPT="$(cat {shlex.quote(RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH)} 2>/dev/null || true)"
cd "${{AGENT_WORKDIR:-{workdir}}}"
rlm "$(cat {shlex.quote(instruction_path)})"
"""

    @classmethod
    def _metric(cls, state: ConfigMap, key: str) -> float:
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


def load_harness(config: RLMConfig) -> RLM:
    return RLM(config=config)


def rlm_checkout_path(
    rlm_repo_url: str,
    rlm_repo_ref: str,
    local_checkout: str | None = None,
    gh_token: str | None = None,
) -> Path:
    return rlm_checkout_loader(
        local_checkout=local_checkout,
        rlm_repo_url=rlm_repo_url,
        rlm_repo_ref=rlm_repo_ref,
        gh_token=gh_token,
    )()


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


def rlm_summarize_at_tokens(
    state: State,
    value: int,
) -> str:
    _ = state
    if value <= 0:
        raise ValueError("summarize_at_tokens must be positive.")
    return str(value)


def rlm_tool_skills_archive(state: State, runtime: Runtime) -> str:
    harness = runtime.harness
    if not isinstance(harness, RLM):
        raise TypeError("rlm_tool_skills_archive requires an RLM harness runtime.")
    tool_defs = runtime.tool_defs(state) or []
    if not tool_defs:
        return ""
    tools = runtime.all_exposed_tools(state)
    used_names: set[str] = set()
    skills_dir = harness.skills_dir()
    if skills_dir is not None:
        root = Path(skills_dir) if isinstance(skills_dir, str) else skills_dir
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
                tool_def.get("description") or f"Call the {tool_name} verifier tool."
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
            required = set(cast(list[str], parameters.get("required") or []))
            type_names = {
                "array": "list",
                "boolean": "bool",
                "integer": "int",
                "null": "None",
                "number": "float",
                "object": "dict",
                "string": "str",
            }
            typed_parameters = all(
                name.isidentifier()
                and not name.startswith("_")
                and not keyword.iskeyword(name)
                and name
                not in {
                    "arguments",
                    "kwargs",
                }
                for name in properties
            )
            if typed_parameters:
                signature_parts: list[str] = []
                argument_lines = ["arguments = {}"]
                for name, raw_schema in sorted(
                    properties.items(),
                    key=lambda item: (
                        "default" in cast(ConfigData, item[1])
                        or item[0] not in required
                    ),
                ):
                    field_schema = cast(ConfigData, raw_schema)
                    raw_type = field_schema.get("type")
                    if isinstance(raw_type, str):
                        annotation_parts = [type_names.get(raw_type, "object")]
                    elif isinstance(raw_type, list):
                        annotation_parts = [
                            type_names.get(str(item), "object") for item in raw_type
                        ]
                    else:
                        annotation_parts = ["object"]
                    annotation = " | ".join(dict.fromkeys(annotation_parts))
                    if name not in required and "default" not in field_schema:
                        if "None" not in annotation_parts:
                            annotation = f"{annotation} | None"
                        signature_parts.append(f"{name}: {annotation} = None")
                        argument_lines.append(f"if {name} is not None:")
                        argument_lines.append(f"    arguments[{name!r}] = {name}")
                    elif "default" in field_schema:
                        default = field_schema["default"]
                        if default is None and "None" not in annotation_parts:
                            annotation = f"{annotation} | None"
                        signature_parts.append(f"{name}: {annotation} = {default!r}")
                        argument_lines.append(f"arguments[{name!r}] = {name}")
                    else:
                        signature_parts.append(f"{name}: {annotation}")
                        argument_lines.append(f"arguments[{name!r}] = {name}")
                signature_parts.append("**kwargs")
                signature = ", ".join(signature_parts)
                argument_lines.append("arguments.update(kwargs)")
                argument_source = textwrap.indent("\n".join(argument_lines), " " * 20)
                call_example = f'result = await {skill_name}(argument_name="value")'
            else:
                signature = "arguments: dict | None = None, **kwargs"
                argument_source = textwrap.indent(
                    "arguments = {**(arguments or {}), **kwargs}", " " * 20
                )
                call_example = (
                    f"result = await {skill_name}({{'argument_name': 'value'}})"
                )
            local_tool_payload: str | None = None
            tool = tools.get(tool_name)
            owner = runtime.tool_owner(tool_name, state)
            if owner is not None and owner.sandbox is None and callable(tool):
                try:
                    tool_signature = inspect.signature(tool)
                except (TypeError, ValueError):
                    tool_signature = None
                if tool_signature is not None and not any(
                    binding_key.partition(".")[0] == tool_name
                    for binding_key in owner.bindings
                ):
                    hidden_args = runtime.hidden_tool_args(tool_name, state)
                    if not any(
                        arg_name in tool_signature.parameters
                        for arg_name in hidden_args
                    ):
                        try:
                            import dill

                            local_tool_payload = base64.b64encode(
                                dill.dumps(tool)
                            ).decode()
                        except Exception:
                            local_tool_payload = None
            if local_tool_payload is None:
                module_imports = "os, requests"
                tool_setup = ""
                dependencies = ["requests", "rlm"]
                call_source = textwrap.indent(
                    textwrap.dedent(
                        f"""\
                        base = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
                        if not base:
                            raise RuntimeError("No Verifiers endpoint URL is configured.")
                        api_key = (
                            os.environ.get("OPENAI_API_KEY")
                            or os.environ.get("ANTHROPIC_API_KEY")
                            or "intercepted"
                        )
                        # Runtime-bound tools fall back to the verifier endpoint.
                        response = requests.post(
                            f"{{base.rsplit('/v1', 1)[0].rstrip('/')}}/vf/tools/" + {tool_name!r},
                            json={{"arguments": arguments}},
                            headers={{"Authorization": f"Bearer {{api_key}}"}},
                            timeout=300,
                        )
                        if not response.content:
                            response.raise_for_status()
                            return None
                        payload = response.json()
                        if "error" in payload:
                            raise RuntimeError(str(payload["error"]))
                        response.raise_for_status()
                        return payload.get("result")
                        """
                    ),
                    " " * 20,
                )
            else:
                module_imports = "base64, dill, inspect"
                tool_setup = (
                    f"TOOL = dill.loads(base64.b64decode({local_tool_payload!r}))"
                )
                dependencies = ["dill", "rlm"]
                call_source = textwrap.indent(
                    textwrap.dedent(
                        """\
                        result = TOOL(**arguments)
                        if inspect.isawaitable(result):
                            return await result
                        return result
                        """
                    ),
                    " " * 20,
                )
            module = textwrap.dedent(
                f"""\
                import {module_imports}


                TOOL_ALLOWED_ARGUMENTS = {allowed_arguments!r}
                {tool_setup}


                async def run({signature}) -> object:
                    {json.dumps(description)}
{argument_source}
                    if TOOL_ALLOWED_ARGUMENTS is not None:
                        arguments = {{
                            key: arguments[key]
                            for key in TOOL_ALLOWED_ARGUMENTS
                            if key in arguments
                        }}
{call_source}
                """
            )
            distribution_name = skill_name.replace("_", "-")
            files = {
                f"{skill_name}/SKILL.md": f"""# {skill_name}

{description}

This skill calls `{tool_name}`.

Call it with tool arguments:

```python
{call_example}
```

Tool schema:

```json
{schema}
```
""",
                f"{skill_name}/{DEFAULT_RLM_TOOL_SKILL_MARKER}": "1\n",
                f"{skill_name}/pyproject.toml": textwrap.dedent(
                    f"""\
                    [project]
                    name = "rlm-skill-{distribution_name}"
                    version = "0.0.0"
                    dependencies = {json.dumps(dependencies)}

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


def keep_only_parent_rlm_steps(step: object, state: State, headers: ConfigMap) -> bool:
    _ = step, state
    return str(headers.get("x-rlm-depth", "0")) == "0"


async def rlm_sub_llm_call_count(task: Task, state: State) -> float:
    _ = task
    return RLM._metric(state, "sub_llm_call_count")


async def rlm_sub_llm_total_turns(task: Task, state: State) -> float:
    _ = task
    return RLM._metric(state, "sub_llm_total_turns")


async def rlm_sub_llm_total_tool_calls(task: Task, state: State) -> float:
    _ = task
    return RLM._metric(state, "sub_llm_total_tool_calls")
