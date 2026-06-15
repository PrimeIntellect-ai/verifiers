"""`python -m verifiers.v1.toolserver` — the generic launcher for a vf-native server class.

Reads the class ref (`VF_SERVER_REF`, `"module:Class"`) and its serialized config
(`VF_SERVER`), rebuilds the instance, and serves its MCP server. Used by `server_to_tools` for
the `command` launch on a host runtime; the sandbox launch generates an equivalent PEP 723
uv-script that calls `serve_server` the same way.
"""

import importlib
import os

from verifiers.v1.tools import serve_server


def main() -> None:
    mod, qual = os.environ["VF_SERVER_REF"].split(":")
    cls = getattr(importlib.import_module(mod), qual)
    serve_server(cls.model_validate_json(os.environ["VF_SERVER"]))


if __name__ == "__main__":
    main()
