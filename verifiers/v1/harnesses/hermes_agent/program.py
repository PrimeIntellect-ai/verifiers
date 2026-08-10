# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["hermes-agent[acp,mcp]=={version}"]
# ///
"""Start Hermes Agent's native ACP server."""

import os

from hermes_cli.models import detect_provider_for_model
from hermes_cli.providers import determine_api_mode

model = os.environ["HERMES_INFERENCE_MODEL"].rsplit("/", 1)[-1]
provider, _ = detect_provider_for_model(model, "auto") or ("auto", model)
os.environ.setdefault("HERMES_INTERCEPT_TRANSPORT", determine_api_mode(provider))

from acp_adapter.entry import main

if __name__ == "__main__":
    main()
