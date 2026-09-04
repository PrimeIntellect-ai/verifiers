# /// script
# requires-python = ">=3.10"
# dependencies = ["mini-swe-agent=={version}", "litellm[proxy]==1.89.2", "httpx"]
# ///

import sys

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.exceptions import Submitted
from minisweagent.run.mini import app

# {tool_interception}

tool_interception_url = ""
tool_interception_secret_bytes = 0
app_arguments = [sys.argv[0]]
for argument in sys.argv[1:]:
    if argument.startswith("--tool-interception-url="):
        tool_interception_url = argument.partition("=")[2]
    elif argument.startswith("--tool-interception-secret-bytes="):
        tool_interception_secret_bytes = int(argument.partition("=")[2])
    else:
        app_arguments.append(argument)
sys.argv = app_arguments

tool_secret = read_tool_secret(tool_interception_secret_bytes, "Mini-SWE")  # noqa: F821 - injected runtime client
tool_interceptor = (
    ToolInterceptionClient(tool_interception_url, tool_secret)  # noqa: F821 - injected runtime client
    if tool_interception_url
    else None
)


class InterceptingAgent(InteractiveAgent):
    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        self._ask_confirmation_or_interrupt([action["command"] for action in actions])
        observations = []
        submitted = None
        for action in actions:
            tool_message = {
                "role": "tool",
                "tool_call_id": action["tool_call_id"],
                "content": "",
                "name": "bash",
            }
            decision = tool_interceptor.call("before", tool_message)
            if decision["action"] == "rewrite":
                observations.append(decision["message"])
                continue
            if submitted is None:
                try:
                    output = self.env.execute(action)
                except Submitted as error:
                    submitted = error
                    output = {
                        "output": "",
                        "returncode": -1,
                        "exception_info": "action was not executed",
                    }
            else:
                output = {
                    "output": "",
                    "returncode": -1,
                    "exception_info": "action was not executed",
                }
            single_message = {
                **message,
                "extra": {**message.get("extra", {}), "actions": [action]},
            }
            observation = self.model.format_observation_messages(
                single_message, [output], self.get_template_vars()
            )[0]
            observation["name"] = "bash"
            decision = tool_interceptor.call("after", observation)
            observations.append(
                decision["message"] if decision["action"] == "rewrite" else observation
            )
        result = self.add_messages(*observations)
        if submitted is not None:
            self._check_for_new_task_or_submit(submitted)
        return result


try:
    app()
finally:
    if tool_interceptor is not None:
        tool_interceptor.close()
