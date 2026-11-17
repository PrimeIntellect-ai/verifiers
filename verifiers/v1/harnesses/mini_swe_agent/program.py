# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["mini-swe-agent=={version}", "litellm[proxy]==1.106.6"]
# ///

import sys
from pathlib import Path

from minisweagent.run.mini import app

# The harness passes the task as a file. Expanding it in sys.argv keeps it out of
# this process's /proc/<pid>/cmdline, which stays as exec'd.
index = sys.argv.index("--task-file")
task_file = Path(sys.argv[index + 1])
sys.argv[index : index + 2] = ["--task", task_file.read_text(encoding="utf-8")]
task_file.unlink()

app()
