"""`python -m verifiers.v1.toolserver` — the generic launcher for a vf-native server class.

Reads the class refs (`VF_SERVER_REF` / `VF_CONFIG_REF` / `VF_TASK_REF`, each `"module:Class"`)
and the serialized `config` + `task` (`VF_CONFIG` / `VF_TASK`), rebuilds `server_cls(config)`,
and serves its MCP server with `setup(task)`. Used by `server_to_tools` for the `command` launch
on a host runtime; the sandbox launch generates an equivalent PEP 723 uv-script that does the same.
"""

import importlib
import os

from verifiers.v1.tools import serve_server


def _ref(var: str) -> type:
    mod, qual = os.environ[var].split(":")
    return getattr(importlib.import_module(mod), qual)


def main() -> None:
    server = _ref("VF_SERVER_REF")(_ref("VF_CONFIG_REF").model_validate_json(os.environ["VF_CONFIG"]))
    task = _ref("VF_TASK_REF").model_validate_json(os.environ["VF_TASK"])
    serve_server(server, task)


if __name__ == "__main__":
    main()
