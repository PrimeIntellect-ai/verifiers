"""The default agent's program: a simple openai chat loop with a bash tool.

`DefaultAgent` stages this single file into the runtime and runs it as
`agent "<instruction>"`. It loops the model with one built-in `bash` tool —
running each requested command locally (in its runtime) and feeding the output
back — until the model answers without a tool call. Model calls go to the
interception server (`OPENAI_BASE_URL`/`OPENAI_API_KEY`), which records the turns.
"""

import json
import os
import subprocess
import sys

from openai import OpenAI

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command and return its combined stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."}
            },
            "required": ["command"],
        },
    },
}


def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True, timeout=60
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"error: {e}"


def main() -> None:
    client = OpenAI()  # OPENAI_BASE_URL + OPENAI_API_KEY come from the environment
    model = os.environ["OPENAI_MODEL"]
    messages = [{"role": "user", "content": sys.argv[1]}]
    while True:
        message = (
            client.chat.completions.create(
                model=model, messages=messages, tools=[BASH_TOOL]
            )
            .choices[0]
            .message
        )
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            break
        for call in message.tool_calls:
            command = json.loads(call.function.arguments).get("command", "")
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": run_bash(command)}
            )


if __name__ == "__main__":
    main()
