# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["mini-swe-agent=={version}", "litellm[proxy]==1.110.0"]
# ///

import sys
from pathlib import Path

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.run.mini import app

# LiteLLM nests OpenRouter `reasoning` and `reasoning_details` under
# `provider_specific_fields` and sends them nested on replay; promote them back
# to top-level wire keys without altering `reasoning_content`-only turns.
_orig_prepare = LitellmModel._prepare_messages_for_api


def _prepare_with_reasoning_details(self, messages):
    prepared = _orig_prepare(self, messages)
    for msg in prepared:
        psf = msg.get("provider_specific_fields")
        if msg.get("role") != "assistant" or not isinstance(psf, dict):
            continue
        details = psf.get("reasoning_details")
        reasoning = psf.get("reasoning")
        if not details and not reasoning:
            continue
        remaining = {
            k: v for k, v in psf.items() if k not in ("reasoning_details", "reasoning")
        }
        if remaining:
            msg["provider_specific_fields"] = remaining
        else:
            msg.pop("provider_specific_fields", None)
        if isinstance(details, list) and details:
            msg["reasoning_details"] = details
        if isinstance(reasoning, str) and reasoning:
            msg["reasoning"] = reasoning
            msg.pop("reasoning_content", None)  # LiteLLM's copy of `reasoning`
    return prepared


LitellmModel._prepare_messages_for_api = _prepare_with_reasoning_details

# The harness passes the task as a file. Expanding it in sys.argv keeps it out of
# this process's /proc/<pid>/cmdline, which stays as exec'd.
index = sys.argv.index("--task-file")
task_file = Path(sys.argv[index + 1])
sys.argv[index : index + 2] = ["--task", task_file.read_text(encoding="utf-8")]
task_file.unlink()

agent = app(standalone_mode=False)
if isinstance(agent, int):
    sys.exit(agent)

messages = getattr(agent, "messages", None)
if (
    messages
    and messages[-1].get("extra", {}).get("exit_status") == "RepeatedFormatError"
):
    sys.exit(int("{repeated_format_error_exit_code}"))
