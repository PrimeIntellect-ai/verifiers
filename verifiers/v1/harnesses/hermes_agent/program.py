# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = []
# ///
"""Start Hermes Agent's native ACP server in its installed project environment."""

import json
import os
import urllib.request

from hermes_cli.models import detect_provider_for_model
from hermes_cli.providers import determine_api_mode

model = os.environ["HERMES_INFERENCE_MODEL"].rsplit("/", 1)[-1]
provider, _ = detect_provider_for_model(model, "auto") or ("auto", model)
os.environ.setdefault("HERMES_INTERCEPT_TRANSPORT", determine_api_mode(provider))

# Hermes' plugin pre-tool hooks intentionally fail open. Wrap its common native
# dispatcher instead: both sequential and concurrent execution pass this point,
# and a failed gate can never fall back to the local callable.
if gate_url := os.environ.get("VF_TOOL_GATE_URL"):
    from agent import tool_executor

    original_dispatch = tool_executor._dispatch_authorized_once
    gate_secret = os.environ.pop("VF_TOOL_GATE_SECRET")
    remote_only = os.environ.get("VF_REMOTE_TOOLS_ONLY") == "1"

    def gated_dispatch(agent, state, ref, **kwargs):
        request = urllib.request.Request(
            gate_url,
            data=json.dumps(
                {"tool_call_id": ref.call_id, "name": ref.name, "arguments": ref.args}
            ).encode(),
            headers={
                "Authorization": f"Bearer {gate_secret}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=None) as response:
            verdict = json.load(response)
        if verdict["action"] == "deny":
            if kwargs.get("begin_execution") is not None:
                kwargs["begin_execution"]()
            state.blocked = True
            content = verdict["message"]["content"]
            return content if isinstance(content, str) else json.dumps(content)
        if verdict["action"] != "allow" or remote_only:
            raise RuntimeError("Split tool gate did not return a remote result")
        return original_dispatch(agent, state, ref, **kwargs)

    tool_executor._dispatch_authorized_once = gated_dispatch

    # The default ACP factory launches a separate gateway, which would not
    # inherit this dispatcher. Select its native in-process execution owner so
    # every sequential and concurrent call crosses the installed gate.
    from acp_adapter import server

    class GatedACPAgent(server.HermesACPAgent):
        def __new__(cls, session_manager=None):
            return object.__new__(cls)

    server.HermesACPAgent = GatedACPAgent

from acp_adapter.entry import main

if __name__ == "__main__":
    main()
