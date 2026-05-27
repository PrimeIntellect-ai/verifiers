import base64
import hashlib
import io
import inspect
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
from verifiers.v1.config import SandboxConfig, sandbox_config_mapping
from verifiers.v1.harness import Harness
from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset
from verifiers.v1.types import ConfigData, ConfigMap, ProgramCommand, ProgramValue
from verifiers.v1.utils.program_utils import int_config
from verifiers.v1.utils.prompt_utils import task_text

from .command import command_program
from .configs import (
    RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH,
    RLMConfig,
)

DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_SKILLS_PATH = "/task/rlm-skills"
DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH = "/tmp/vf-rlm-tool-skills.tar.gz.b64"
DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME = ".vf-generated-tool-skills"
DEFAULT_RLM_TOOL_SKILL_MARKER = ".vf-generated-tool-skill"
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
            explicit_sandbox_options = (
                sandbox_config_mapping(sandbox_config, fill_defaults=False) or {}
            )
            if explicit_sandbox_options.get("setup_timeout") is not None:
                setup_timeout = int_config(
                    explicit_sandbox_options, "setup_timeout", setup_timeout
                )
            sandbox_config = {
                "workdir": harness_config.workdir,
                "command_timeout": max(harness_config.rlm_exec_timeout + 120, 600),
                **sandbox_options,
                "setup_timeout": setup_timeout,
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
        self._explicit_skills = self._skills_dir is not None
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
                "bash -lc "
                + shlex.quote(
                    f"""
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
        program.setdefault("setup_timeout", setup_timeout)
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
        if not self._explicit_skills:
            upload_dirs = taskset.get_upload_dirs()
            if not isinstance(upload_dirs, Mapping):
                raise TypeError("Taskset.get_upload_dirs() must return a mapping.")
            self._skills_dir = cast(ProgramDir | None, upload_dirs.get("skills"))
            if not isinstance(self.program, Mapping):
                raise TypeError("RLM program must be a mapping.")
            program = dict(cast(ConfigMap, self.program))
            dirs = dict(cast(ConfigMap, program.get("dirs") or {}))
            if self._skills_dir is None:
                dirs.pop(DEFAULT_RLM_SKILLS_PATH, None)
            else:
                dirs[DEFAULT_RLM_SKILLS_PATH] = self._skills_dir
            if dirs:
                program["dirs"] = dirs
            else:
                program.pop("dirs", None)
            self.program = program
        super().attach_taskset(taskset)
        self._program = self.compile_program(self.program)

    def vf_tool_skills_archive(self, state: State) -> str:
        tool_defs = self.runtime.tool_defs(state) or []
        if not tool_defs:
            return ""
        tools = self.runtime.all_exposed_tools(state)
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
                            signature_parts.append(
                                f"{name}: {annotation} = {default!r}"
                            )
                            argument_lines.append(f"arguments[{name!r}] = {name}")
                        else:
                            signature_parts.append(f"{name}: {annotation}")
                            argument_lines.append(f"arguments[{name!r}] = {name}")
                    signature_parts.append("**kwargs")
                    signature = ", ".join(signature_parts)
                    argument_lines.append("arguments.update(kwargs)")
                    argument_source = textwrap.indent(
                        "\n".join(argument_lines), " " * 24
                    )
                    call_example = f'result = await {skill_name}(argument_name="value")'
                else:
                    signature = "arguments: dict | None = None, **kwargs"
                    argument_source = textwrap.indent(
                        "arguments = {**(arguments or {}), **kwargs}", " " * 24
                    )
                    call_example = (
                        f"result = await {skill_name}({{'argument_name': 'value'}})"
                    )
                local_tool_payload: str | None = None
                tool = tools.get(tool_name)
                owner = self.runtime.tool_owner(tool_name, state)
                if owner is not None and owner.sandbox is None and callable(tool):
                    try:
                        tool_signature = inspect.signature(tool)
                    except (TypeError, ValueError):
                        tool_signature = None
                    if tool_signature is not None and not any(
                        binding_key.partition(".")[0] == tool_name
                        for binding_key in owner.bindings
                    ):
                        hidden_args = self.runtime.hidden_tool_args(tool_name, state)
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
                        " " * 24,
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
                        " " * 24,
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
