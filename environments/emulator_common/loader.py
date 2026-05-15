import json
from pathlib import Path

from verifiers.v1.types import ConfigData

from .sandbox import build_sandbox_config
from .task_schema import EmulatorManifest


def load_manifest(path: str | Path) -> EmulatorManifest:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as f:
        data = json.load(f)
    manifest = EmulatorManifest.from_mapping(data)
    if manifest.module_name != manifest_path.parent.parent.name:
        raise ValueError(
            "manifest module_name must match its environment directory: "
            f"{manifest.module_name!r} != {manifest_path.parent.parent.name!r}"
        )
    return manifest


def build_instruction(manifest: EmulatorManifest) -> str:
    public = "\n".join(f"- {name}" for name in manifest.public_verification)
    hidden = "\n".join(f"- {name}" for name in manifest.hidden_verification)
    requirements = "\n".join(f"- {item}" for item in manifest.requirements)
    concepts = ", ".join(manifest.core_concepts)
    exposed = "\n".join(f"- {item}" for item in manifest.exposed_api)
    success = "\n".join(f"- {item}" for item in manifest.success_criteria)
    framebuffer = manifest.framebuffer or {"width": 0, "height": 0}
    return f"""\
{manifest.base_prompt}

You are working in /workspace. Implement this as a Rust crate.

Benchmark level: {manifest.level}
System: {manifest.display_name}
Difficulty: {manifest.difficulty}
Core concepts: {concepts}

Functional requirements:
{requirements}

Required public API:
{exposed or "- Use the common Emulator API from src/lib.rs."}

Framebuffer contract:
- Width: {framebuffer.get("width", 0)}
- Height: {framebuffer.get("height", 0)}
- framebuffer() must return deterministic, row-major pixels.

Public verification targets:
{public}

Hidden verification categories:
{hidden}

Success criteria:
{success}

You may add dependencies, modules, tests, and build scripts as needed. Keep the
emulator deterministic: identical ROM bytes, input events, and cycle budgets
must produce identical framebuffer hashes, traces, and observable outputs.
"""


def manifest_to_rows(
    manifest: EmulatorManifest,
    *,
    max_tasks: int | None = None,
    cpu_cores: int | None = None,
    memory_gb: int | None = None,
    network_access: bool = True,
    sandbox_timeout_minutes: int | None = None,
) -> list[ConfigData]:
    info = manifest.to_info()
    instruction = build_instruction(manifest)
    row = {
        "example_id": 0,
        "task_id": f"{manifest.environment_id}__implementation",
        "question": instruction,
        "instruction": instruction,
        "prompt": [{"role": "user", "content": instruction}],
        "answer": "pass",
        "info": info,
        "sandbox": build_sandbox_config(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            network_access=network_access,
            timeout_minutes=sandbox_timeout_minutes,
        ),
    }
    rows = [row]
    return rows[:max_tasks] if max_tasks else rows
