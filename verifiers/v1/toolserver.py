"""Generic launcher for a vf-native tool/user server: `python -m verifiers.v1.toolserver`.

The framework starts a `Toolset`/`User` by running this module in the server's runtime (host or
sandbox). It reconstructs the authored server from the environment the framework set: `VF_SERVER`
and `VF_CONFIG_CLS` name the server class and its config class as `module:qualname`, `VF_CONFIG`
carries the config JSON, and `VF_TASK_CLS`/`VF_TASK` carry this rollout's task (absent for a
`shared`, task-agnostic server — then `setup` gets `None`). It imports the *real* classes — so
`import verifiers`, sibling imports, and module-level globals all resolve, and the task validates
against its actual (extra-bearing) subclass — rebuilds the instance, and serves it over MCP on
`MCP_PORT`. Nothing is inlined: the server's own module is installed/importable in the runtime
(ambient on the host; uploaded + `uv pip install`ed in a sandbox)."""

from __future__ import annotations

import importlib
import os

from verifiers.v1.tools import serve_server


def _load(ref: str) -> object:
    """Resolve a `module:qualname` reference (e.g. `glossary_v1:GlossaryToolset`) to the object."""
    module_name, _, qualname = ref.partition(":")
    obj: object = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def main() -> None:
    cls = _load(os.environ["VF_SERVER"])
    config_cls = _load(os.environ["VF_CONFIG_CLS"])
    config = config_cls.model_validate_json(os.environ["VF_CONFIG"])
    task = None  # a `shared` server is task-agnostic — the framework ships no task
    if "VF_TASK" in os.environ:
        task = _load(os.environ["VF_TASK_CLS"]).model_validate_json(os.environ["VF_TASK"])
    serve_server(cls(config), task)


if __name__ == "__main__":
    main()
