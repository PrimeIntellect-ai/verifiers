# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = ["mini-swe-agent=={version}", "litellm[proxy]==1.106.6"]
# ///

import json
import sys
from pathlib import Path

from minisweagent.run.mini import app

# The arguments, the model key and task among them, come in a file that is deleted
# once read, never argv: the box's `ps` would show them to the agent.
path = Path(sys.argv[1])
args = json.loads(path.read_bytes())
path.unlink()
app(args=args)
