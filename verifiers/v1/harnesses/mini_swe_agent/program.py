# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["mini-swe-agent=={version}", "litellm[proxy]==1.110.0"]
# ///

import os
import sys
from pathlib import Path

from minisweagent.environments.local import LocalEnvironment
from minisweagent.run.mini import app

# REMOTE_TOOL_CLIENT


remote_url = os.environ["VF_REMOTE_TOOL_URL"]


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

# The harness passes the task as a file. Expanding it in sys.argv keeps it out of
# this process's /proc/<pid>/cmdline, which stays as exec'd.
index = sys.argv.index("--task-file")
task_file = Path(sys.argv[index + 1])
sys.argv[index : index + 2] = ["--task", task_file.read_text(encoding="utf-8")]
task_file.unlink()

app()
