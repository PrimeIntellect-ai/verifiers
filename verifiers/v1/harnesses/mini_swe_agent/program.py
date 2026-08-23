# /// script
# requires-python = ">=3.10"
# dependencies = ["mini-swe-agent=={version}", "litellm[proxy]==1.89.2", "httpx"]
# ///

import sys

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.exceptions import Submitted
from minisweagent.run.mini import app

# {toolInterception}

toolInterceptionUrl = ""
toolInterceptionSecretBytes = 0
appArguments = [sys.argv[0]]
for argument in sys.argv[1:]:
    if argument.startswith("--tool-interception-url="):
        toolInterceptionUrl = argument.partition("=")[2]
    elif argument.startswith("--tool-interception-secret-bytes="):
        toolInterceptionSecretBytes = int(argument.partition("=")[2])
    else:
        appArguments.append(argument)
sys.argv = appArguments

toolSecret = readToolSecret(toolInterceptionSecretBytes, "Mini-SWE")  # noqa: F821 - injected runtime client
toolInterceptor = (
    ToolInterceptionClient(toolInterceptionUrl, toolSecret)  # noqa: F821 - injected runtime client
    if toolInterceptionUrl
    else None
)


class InterceptingAgent(InteractiveAgent):
    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        self._ask_confirmation_or_interrupt([action["command"] for action in actions])
        observations = []
        submitted = None
        for action in actions:
            toolMessage = {
                "role": "tool",
                "tool_call_id": action["tool_call_id"],
                "content": "",
                "name": "bash",
            }
            decision = toolInterceptor.call("before", toolMessage)
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
            singleMessage = {
                **message,
                "extra": {**message.get("extra", {}), "actions": [action]},
            }
            observation = self.model.format_observation_messages(
                singleMessage, [output], self.get_template_vars()
            )[0]
            observation["name"] = "bash"
            decision = toolInterceptor.call("after", observation)
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
    if toolInterceptor is not None:
        toolInterceptor.close()
