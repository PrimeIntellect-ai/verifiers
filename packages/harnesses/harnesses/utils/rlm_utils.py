import base64
import io
import inspect
import json
import keyword
import os
import re
import tarfile
import textwrap
from collections.abc import Callable
from collections.abc import Mapping
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

from verifiers.envs.experimental.utils.git_checkout_cache import (
    resolve_git_checkout,
    validate_git_checkout,
)
from verifiers.v1.runtime import Runtime
from verifiers.v1.state import State
from verifiers.v1.types import ConfigData

DEFAULT_RLM_CHECKOUT_PATH = "/tmp/rlm-checkout"
DEFAULT_RLM_SKILLS_PATH = "/task/rlm-skills"
DEFAULT_RLM_TOOL_SKILLS_ARCHIVE_PATH = "/tmp/vf-rlm-tool-skills.tar.gz.b64"
DEFAULT_RLM_TOOL_SKILLS_MANIFEST_NAME = ".vf-generated-tool-skills"
DEFAULT_RLM_TOOL_SKILL_MARKER = ".vf-generated-tool-skill"
DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT = (
    Path.home() / ".cache" / "verifiers" / "rlm-checkouts"
)
REQUIRED_RLM_CHECKOUT_FILES = ("install.sh", "pyproject.toml")


def rlm_checkout_path(
    rlm_repo_url: str,
    rlm_repo_ref: str,
    local_checkout: str | None = None,
    gh_token_var: str | None = "GH_TOKEN",
) -> Path:
    return rlm_checkout_loader(
        local_checkout=local_checkout,
        rlm_repo_url=rlm_repo_url,
        rlm_repo_ref=rlm_repo_ref,
        gh_token_var=gh_token_var,
    )()


def rlm_checkout_loader(
    local_checkout: str | Path | None,
    rlm_repo_url: str,
    rlm_repo_ref: str,
    gh_token_var: str | None,
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
                gh_token=os.environ.get(gh_token_var) if gh_token_var else None,
                required_files=REQUIRED_RLM_CHECKOUT_FILES,
            )
        return checkout

    return load


def rlm_tool_skills_archive(state: State, runtime: Runtime) -> str:
    tool_defs = runtime.tool_defs(state) or []
    if not tool_defs:
        return ""
    tools = runtime.all_exposed_tools(state)
    used_names: set[str] = set()
    skills_dir = rlm_skills_dir(state, runtime)
    if skills_dir is not None:
        used_names.update(
            child.name for child in skills_dir.iterdir() if child.is_dir()
        )
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


def rlm_skills_dir(state: State, runtime: Runtime) -> Path | Traversable | None:
    from harnesses.rlm import RLM

    _ = state
    harness = runtime.harness
    if not isinstance(harness, RLM):
        raise TypeError("rlm_skills_dir requires an RLM harness runtime.")
    if harness.config.skills is not None:
        return Path(harness.config.skills)
    taskset = runtime.taskset
    if taskset is None:
        return None
    upload_dirs = taskset.get_upload_dirs()
    assert isinstance(upload_dirs, Mapping)
    skills_dir = upload_dirs.get("skills")
    if skills_dir is None:
        return None
    assert isinstance(skills_dir, (Path, Traversable))
    return skills_dir
