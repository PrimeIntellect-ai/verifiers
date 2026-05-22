"""Endpoint-backed RLM skill generation for V1 tools."""

import hashlib
import keyword
import re
import shutil
from collections.abc import Mapping
from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from typing import TypedDict, cast

from verifiers.types import Tool

from ...runtime import Runtime
from ...state import State
from ...task import Task
from ...types import ConfigMap

RLM_SKILLS_CACHE_ROOT = Path.home() / ".cache" / "verifiers" / "rlm-skills"


class SkillParam(TypedDict):
    name: str
    annotation: str
    description: str
    required: bool
    default_literal: str | None


def stage_rlm_tool_skills(
    task: Task,
    state: State,
    runtime: Runtime,
    *,
    source: Path | Traversable | str | None = None,
    cache_root: Path = RLM_SKILLS_CACHE_ROOT,
) -> Path:
    task_id = str(task.get("task_id") or task.get("task_name") or "task")
    key = str(state.get("trajectory_id") or id(state)).replace("/", "_")
    target = cache_root / task_id / key
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    if source is not None:
        copy_skill_source(source, target)
    existing = {path.name for path in target.iterdir() if path.is_dir()}
    for tool in runtime.tool_defs(state) or []:
        skill_name = rlm_skill_name(tool.name)
        if skill_name not in existing:
            write_tool_skill(target / skill_name, tool, skill_name)
    return target


def copy_skill_source(source: Path | Traversable | str, target: Path) -> None:
    if isinstance(source, str):
        source = Path(source)
    with resources.as_file(source) as path:
        shutil.copytree(path, target, dirs_exist_ok=True)


def rlm_skill_name(tool_name: str) -> str:
    name = re.sub(r"\W", "_", tool_name)
    if not name or name[0].isdigit():
        name = f"tool_{name}"
    if keyword.iskeyword(name):
        name = f"{name}_tool"
    if name == tool_name:
        return name
    digest = hashlib.sha1(tool_name.encode()).hexdigest()[:8]
    return f"{name}_{digest}"


def write_tool_skill(skill_dir: Path, tool: Tool, skill_name: str) -> None:
    src_dir = skill_dir / "src" / skill_name
    src_dir.mkdir(parents=True, exist_ok=True)
    params = params_from_schema(tool.parameters)
    summary = tool.description
    arg_descs = {param["name"]: param["description"] for param in params}
    (src_dir / "__init__.py").write_text(_INIT_MODULE.format(skill_name=skill_name))
    (src_dir / f"{skill_name}.py").write_text(
        build_skill_module(tool.name, skill_name, params, summary, arg_descs)
    )
    (skill_dir / "pyproject.toml").write_text(
        _PYPROJECT.format(
            skill_name=skill_name, skill_dash=skill_name.replace("_", "-")
        )
    )
    (skill_dir / "SKILL.md").write_text(
        skill_markdown(tool.name, skill_name, params, summary, arg_descs)
    )


def params_from_schema(schema: ConfigMap) -> list[SkillParam]:
    properties = schema.get("properties") if isinstance(schema, Mapping) else {}
    properties = properties if isinstance(properties, Mapping) else {}
    raw_required = schema.get("required") if isinstance(schema, Mapping) else []
    required = (
        {name for name in raw_required if isinstance(name, str)}
        if isinstance(raw_required, list)
        else set()
    )
    params: list[SkillParam] = []
    for name, value in properties.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or keyword.iskeyword(name)
        ):
            continue
        field_schema = cast(ConfigMap, value) if isinstance(value, Mapping) else {}
        annotation = annotation_from_schema(field_schema)
        is_required = name in required
        params.append(
            {
                "name": name,
                "annotation": annotation if is_required else f"{annotation} | None",
                "description": str(field_schema.get("description") or ""),
                "required": is_required,
                "default_literal": None if is_required else "None",
            }
        )
    return params


def annotation_from_schema(schema: ConfigMap) -> str:
    value = schema.get("type")
    if isinstance(value, list):
        value = next((item for item in value if item != "null"), None)
    if value == "integer":
        return "int"
    if value == "number":
        return "float"
    if value == "boolean":
        return "bool"
    if value == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            return f"list[{annotation_from_schema(cast(ConfigMap, items))}]"
        return "list[str]"
    if value == "object":
        return "dict"
    return "str"


def build_skill_module(
    tool_name: str,
    skill_name: str,
    params: list[SkillParam],
    summary: str,
    arg_descs: dict[str, str],
) -> str:
    signature_parts = []
    for param in params:
        part = f"{param['name']}: {param['annotation']}"
        if not param["required"]:
            part += f" = {param['default_literal']}"
        signature_parts.append(part)
    arguments = (
        "{" + ", ".join(f"{param['name']!r}: {param['name']}" for param in params) + "}"
    )
    docstring = skill_docstring(tool_name, params, summary, arg_descs)
    return _SKILL_MODULE.format(
        tool_name=tool_name,
        skill_name=skill_name,
        signature=", ".join(signature_parts),
        arguments=arguments,
        docstring=repr(docstring),
    )


def skill_docstring(
    tool_name: str,
    params: list[SkillParam],
    summary: str,
    arg_descs: dict[str, str],
) -> str:
    lines = [summary or f"Call the {tool_name} V1 tool."]
    if arg_descs:
        lines.extend(["", "Args:"])
        for param in params:
            desc = arg_descs.get(param["name"], "")
            lines.append(f"    {param['name']}: {desc}".rstrip())
    return "\n".join(lines).strip()


def skill_markdown(
    tool_name: str,
    skill_name: str,
    params: list[SkillParam],
    summary: str,
    arg_descs: dict[str, str],
) -> str:
    lines = [f"# {skill_name}", ""]
    if summary:
        lines.extend([summary, ""])
    if skill_name != tool_name:
        lines.extend([f"Calls the V1 tool `{tool_name}`.", ""])
    lines.append("Parameters:")
    for param in params:
        optional = "" if param["required"] else " (optional)"
        desc = arg_descs.get(param["name"], "")
        lines.append(
            f"- `{param['name']}` ({param['annotation']}){optional}: {desc}".rstrip(
                ": "
            )
        )
    lines.extend(
        ["", "From IPython:", "```python", f"await {skill_name}(...)", "```", ""]
    )
    return "\n".join(lines)


_PYPROJECT = """\
[project]
name = "rlm-skill-{skill_dash}"
version = "0.1.0"
requires-python = ">=3.10"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{skill_name}"]
"""


_INIT_MODULE = '''\
"""Auto-generated RLM skill: {skill_name}."""

from .{skill_name} import {skill_name}, run

__all__ = ["{skill_name}", "run"]
'''


_SKILL_MODULE = '''\
"""Auto-generated RLM skill: {skill_name}."""

import json
import os
import urllib.error
import urllib.request

TOOL_NAME = {tool_name!r}


def _endpoint_root() -> str:
    root = (
        os.environ.get("VF_ENDPOINT_ROOT_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
    )
    if root:
        return root.rstrip("/")
    base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _call_endpoint(arguments):
    root = _endpoint_root()
    if not root:
        raise RuntimeError("VF endpoint URL is not configured")
    payload = json.dumps({{"arguments": arguments}}).encode()
    token = (
        os.environ.get("VF_ENDPOINT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or "intercepted"
    )
    request = urllib.request.Request(
        f"{{root}}/vf/tools/{{TOOL_NAME}}",
        data=payload,
        headers={{
            "Authorization": f"Bearer {{token}}",
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python",
        }},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")
        raise RuntimeError(body or str(error)) from error
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return result.get("result")


async def {skill_name}({signature}):
    {docstring}
    return _call_endpoint(
        {{key: value for key, value in {arguments}.items() if value is not None}}
    )


run = {skill_name}
'''
