# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["mini-swe-agent=={version}", "litellm[proxy]==1.106.6"]
# ///

import os

from minisweagent.environments.local import LocalEnvironment
from minisweagent.run.mini import app

# REMOTE_TOOL_CLIENT


if remote_url := os.environ.get("VF_REMOTE_TOOL_URL"):

    def execute(self, action, cwd="", *, timeout=None):
        content = remote_tool(  # noqa: F821 - injected shared client
            remote_url,
            os.environ["VF_REMOTE_TOOL_SECRET"],
            "bash",
            {
                "command": action.get("command", ""),
                "cwd": cwd or self.config.cwd or os.environ["VF_EXECUTION_CWD"],
                "env": self.config.env,
                "combine_output": True,
                "timeout": timeout or self.config.timeout,
            },
            action.get("tool_call_id"),
        )
        result = remote_shell_result(content)  # noqa: F821
        output = {
            "output": result["stdout"] + result["stderr"],
            "returncode": result["returncode"],
            "exception_info": result["exception_info"],
        }
        self._check_finished(output)
        return output

    LocalEnvironment.execute = execute

app()
