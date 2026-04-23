#!/usr/bin/env python3
"""Sandbox-side Terminus 2 runner.

The runner is dependency-free on purpose: it talks to the intercepted
OpenAI-compatible endpoint with urllib, controls a tmux shell, and parses the
JSON command format used by Harbor's Terminus 2 prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element, SubElement, indent, tostring
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DEFAULT_JSON_PROMPT_TEMPLATE = """You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }},
    {{
      "keystrokes": "cd project\\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

Command object structure:
- "keystrokes": String containing the exact keystrokes to send to the terminal (required)
- "duration": Number of seconds to wait for the command to complete before the next command will be executed (defaults to 1.0 if not present)

IMPORTANT: The text inside "keystrokes" will be used completely verbatim as keystrokes. Write commands exactly as you want them sent to the terminal:
- You must end every command with a newline (\\n) or it will not execute.
- For special key sequences, use tmux-style escape sequences:
  - C-c for Ctrl+C
  - C-d for Ctrl+D

The "duration" attribute specifies the number of seconds to wait for the command to complete (default: 1.0) before the next command will be executed. On immediate tasks (e.g., cd, ls, echo, cat) set a duration of 0.1 seconds. On commands (e.g., gcc, find, rustc) set a duration of 1.0 seconds. On slow commands (e.g., make, python3 [long running script], wget [file]) set an appropriate duration as you determine necessary.

It is better to set a smaller duration than a longer duration. It is always possible to wait again if the prior output has not finished, by running {{"keystrokes": "", "duration": 10.0}} on subsequent requests to wait longer. Never wait longer than 60 seconds; prefer to poll to see intermediate result status.

Important notes:
- Each command's keystrokes are sent exactly as written to the terminal
- Do not include extra whitespace before or after the keystrokes unless it's part of the intended command
- Extra text before or after the JSON will generate warnings but be tolerated
- The JSON must be valid - use proper escaping for quotes and special characters within strings
- Commands array can be empty if you want to wait without taking action

Task Description:
{instruction}

Current terminal state:
{terminal_state}
"""

SPECIAL_KEYS = {
    "C-c",
    "C-d",
    "C-m",
    "C-j",
    "Enter",
    "KPEnter",
    "^M",
    "^J",
}


class TerminusRunner:
    """Drive the Terminus loop by alternating model calls and tmux commands."""

    def __init__(self, args: argparse.Namespace):
        """Initialize runner state from parsed CLI arguments."""
        self.args = args
        self.session_name = f"terminus_2_{os.getpid()}"
        self.previous_buffer: str | None = None
        self.messages: list[dict[str, str]] = []
        self.pending_completion = False
        self.original_instruction = ""
        self.summarization_count = 0
        self.log_path = Path(args.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        """Emit a structured event to stdout and the runner log file."""
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self.log_path.open("a") as f:
            f.write(line + "\n")
        print(line, flush=True)

    def run_tmux(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
        """Run a tmux command and return its captured result."""
        return subprocess.run(command, text=True, capture_output=True, **kwargs)

    def start_session(self) -> None:
        """Start a detached shell session and pipe pane output to a log."""
        Path(self.args.agent_workdir).mkdir(parents=True, exist_ok=True)
        pane_log = str(self.log_path.with_suffix(".pane"))
        self.run_tmux(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                self.session_name,
                "-x",
                str(self.args.tmux_width),
                "-y",
                str(self.args.tmux_height),
                "-c",
                self.args.agent_workdir,
                "bash --login",
            ],
            check=True,
        )
        self.run_tmux(
            ["tmux", "set-option", "-t", self.session_name, "history-limit", "1000000"],
            check=True,
        )
        self.run_tmux(
            ["tmux", "pipe-pane", "-t", self.session_name, f"cat > {pane_log}"],
            check=True,
        )
        self.log(
            "session_started",
            session=self.session_name,
            workdir=self.args.agent_workdir,
        )

    def stop_session(self) -> None:
        """Terminate the tmux session created for this rollout."""
        self.run_tmux(["tmux", "kill-session", "-t", self.session_name])
        self.log("session_stopped", session=self.session_name)

    def capture_pane(self, entire: bool = False) -> str:
        """Capture either the visible pane or the full tmux history."""
        command = ["tmux", "capture-pane", "-p", "-t", self.session_name]
        if entire:
            command[3:3] = ["-S", "-"]
        result = self.run_tmux(command, check=True)
        return result.stdout or ""

    def incremental_output(self) -> str:
        """Return only new terminal output when possible, else the screen."""
        current = self.capture_pane(entire=True)
        visible = self.capture_pane()
        if self.previous_buffer is None:
            self.previous_buffer = current
            return f"Current Terminal Screen:\n{visible}"

        previous = self.previous_buffer.strip()
        self.previous_buffer = current
        if previous and previous in current:
            new = current[current.index(previous) + len(previous) :].strip()
            if new:
                return f"New Terminal Output:\n{new}"
        return f"Current Terminal Screen:\n{visible}"

    def send_keystrokes(self, keystrokes: str) -> None:
        """Send raw text or tmux special keys into the managed shell.

        Normal text goes through a temporary tmux buffer so multiline commands
        and literal shell input are pasted exactly as the model wrote them.
        """
        if keystrokes in SPECIAL_KEYS:
            self.run_tmux(
                ["tmux", "send-keys", "-t", self.session_name, keystrokes], check=True
            )
            return

        buffer_name = f"terminus_2_{os.getpid()}"
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(keystrokes)
            tmp_path = f.name
        try:
            self.run_tmux(
                ["tmux", "load-buffer", "-b", buffer_name, tmp_path], check=True
            )
            self.run_tmux(
                [
                    "tmux",
                    "paste-buffer",
                    "-b",
                    buffer_name,
                    "-d",
                    "-t",
                    self.session_name,
                ],
                check=True,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def call_chat(self, messages: list[dict[str, str]]) -> str:
        """Call the intercepted OpenAI-compatible chat completions endpoint."""
        base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
        model = os.environ["OPENAI_MODEL"]
        api_key = os.environ.get("OPENAI_API_KEY", "intercepted")
        timeout = float(os.environ.get("OPENAI_TIMEOUT", "3600"))
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": self.args.temperature,
            }
        ).encode()
        request = Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "OpenAI/Python",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
        except HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible request failed: {e.code} {detail}"
            ) from e

        content = payload["choices"][0]["message"].get("content") or ""
        return content

    def call_model(self, prompt: str) -> str:
        """Append a user prompt, summarize if needed, and call the model.

        If a request fails, the runner forces summarization once and retries so
        oversized histories can recover without losing the rollout.
        """
        self.messages.append({"role": "user", "content": prompt})
        self.summarize_if_needed()
        try:
            content = self.call_chat(self.messages)
        except RuntimeError:
            if not self.summarize_if_needed(force=True):
                raise
            content = self.call_chat(self.messages)
        self.messages.append({"role": "assistant", "content": content})
        return content

    def summarize_if_needed(self, force: bool = False) -> bool:
        """Condense the message history when it exceeds the configured budget."""
        if not self.args.enable_summarize:
            return False
        if not force and not should_summarize(
            self.messages,
            self.args.summarization_threshold_chars,
            self.args.summarization_keep_messages,
        ):
            return False
        if len([m for m in self.messages if m["role"] != "system"]) < 3:
            return False

        self.summarization_count += 1
        self.log(
            "summarization_started",
            index=self.summarization_count,
            message_chars=count_message_chars(self.messages),
            forced=force,
        )
        try:
            self.summarize_history()
        except Exception as e:
            self.log(
                "summarization_failed",
                index=self.summarization_count,
                error=str(e),
            )
            return False

        self.log(
            "summarization_complete",
            index=self.summarization_count,
            message_chars=count_message_chars(self.messages),
        )
        return True

    def summarize_history(self) -> None:
        """Replace old history with a three-step agent handoff.

        The current agent summarizes, a simulated next agent asks missing-context
        questions, and the current agent answers before the compacted prompt is
        installed as the new history.
        """
        history = list(self.messages)
        screen = limit_output(self.capture_pane(), self.args.max_output_bytes)
        summary_prompt = build_summary_prompt(self.original_instruction)
        summary = self.call_chat(
            history + [{"role": "user", "content": summary_prompt}]
        )

        question_prompt = build_question_prompt(
            self.original_instruction,
            summary,
            screen,
        )
        questions = self.call_chat([{"role": "user", "content": question_prompt}])

        answer_prompt = (
            "The next agent has a few questions for you. "
            "Answer each of them one by one in detail:\n\n"
            f"{questions}"
        )
        answers = self.call_chat(
            history
            + [
                {"role": "user", "content": summary_prompt},
                {"role": "assistant", "content": summary},
                {"role": "user", "content": answer_prompt},
            ]
        )

        system_messages = [m for m in self.messages if m["role"] == "system"]
        self.messages = system_messages + [
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": questions},
            {"role": "user", "content": build_handoff_prompt(answers)},
        ]

    def execute_commands(self, commands: list[dict[str, Any]]) -> str:
        """Run parsed command keystrokes and return the latest terminal output."""
        for command in commands:
            keystrokes = command["keystrokes"]
            duration = min(max(float(command.get("duration", 1.0)), 0.0), 60.0)
            self.log("command", keystrokes=keystrokes, duration=duration)
            if keystrokes:
                self.send_keystrokes(keystrokes)
            time.sleep(duration)
        return limit_output(self.incremental_output(), self.args.max_output_bytes)

    def run(self) -> int:
        """Run the full Terminus agent loop until completion or max turns.

        The model must return task_complete twice in a row, after seeing the
        latest terminal state, before the runner exits successfully.
        """
        instruction = Path(self.args.instruction_path).read_text()
        mcp_servers = parse_mcp_servers(self.args.mcp_servers_json)
        augmented_instruction = build_augmented_instruction(
            instruction,
            mcp_servers,
            self.args.skills_dir,
        )
        self.original_instruction = augmented_instruction
        template_name = f"terminus-{self.args.parser_name}-plain.txt"
        prompt_template_path = Path(self.args.template_dir) / template_name
        prompt_template = DEFAULT_JSON_PROMPT_TEMPLATE
        if prompt_template_path.exists():
            prompt_template = prompt_template_path.read_text()

        system_prompt_path = Path(self.args.system_prompt_path)
        if system_prompt_path.exists() and system_prompt_path.stat().st_size > 0:
            self.messages.append(
                {"role": "system", "content": system_prompt_path.read_text()}
            )

        self.start_session()
        try:
            prompt = prompt_template.format(
                instruction=augmented_instruction,
                terminal_state=limit_output(
                    self.incremental_output(),
                    self.args.max_output_bytes,
                ),
            )
            for turn in range(self.args.max_turns):
                self.log("turn_started", turn=turn + 1)
                response = self.call_model(prompt)
                parsed = parse_response(response, self.args.parser_name)
                if parsed["error"]:
                    prompt = (
                        f"Previous response had parsing errors:\n{parsed['error']}\n\n"
                        "Please fix these issues and provide a valid JSON response."
                    )
                    continue

                observation = self.execute_commands(parsed["commands"])
                if parsed["task_complete"]:
                    if self.pending_completion:
                        self.log("task_complete_confirmed", turn=turn + 1)
                        return 0
                    self.pending_completion = True
                    prompt = (
                        f"Current terminal state:\n{observation}\n\n"
                        "Are you sure you want to mark the task as complete? "
                        'If so, include "task_complete": true in your JSON response again.'
                    )
                    continue

                self.pending_completion = False
                prompt = observation

            self.log("max_turns_reached", max_turns=self.args.max_turns)
            return 0
        finally:
            self.stop_session()


def extract_json_content(response: str) -> str:
    """Extract the first balanced JSON object from a model response."""
    start = -1
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(response):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start != -1:
                return response[start : i + 1]
    return ""


def extract_tag(content: str, tag: str) -> str | None:
    """Return the trimmed contents of an XML tag, preserving empty tags."""
    full = re.search(f"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    if full:
        return full.group(1).strip()
    if re.search(f"<{tag}\\s*/>", content) or re.search(f"<{tag}></{tag}>", content):
        return ""
    return None


def parse_response(response: str, parser_name: str = "json") -> dict[str, Any]:
    """Parse a model response into commands, completion state, and errors."""
    if parser_name == "xml":
        return parse_xml_response(response)
    if parser_name != "json":
        return {
            "commands": [],
            "task_complete": False,
            "error": f"Unknown parser: {parser_name}",
        }

    json_content = extract_json_content(response)
    if not json_content:
        return {"commands": [], "task_complete": False, "error": "No JSON object found"}
    try:
        payload = json.loads(json_content)
    except json.JSONDecodeError as e:
        return {"commands": [], "task_complete": False, "error": f"Invalid JSON: {e}"}

    missing = [
        field for field in ("analysis", "plan", "commands") if field not in payload
    ]
    if missing:
        return {
            "commands": [],
            "task_complete": False,
            "error": f"Missing required fields: {', '.join(missing)}",
        }
    if not isinstance(payload["commands"], list):
        return {
            "commands": [],
            "task_complete": False,
            "error": "commands must be a list",
        }

    commands = []
    for index, command in enumerate(payload["commands"], start=1):
        if not isinstance(command, dict) or not isinstance(
            command.get("keystrokes"), str
        ):
            return {
                "commands": [],
                "task_complete": False,
                "error": f"Command {index} must include string keystrokes",
            }
        commands.append(
            {
                "keystrokes": command["keystrokes"],
                "duration": command.get("duration", 1.0),
            }
        )

    complete = payload.get("task_complete", False)
    if isinstance(complete, str):
        complete = complete.lower() in {"true", "1", "yes"}
    return {"commands": commands, "task_complete": bool(complete), "error": ""}


def parse_mcp_servers(raw: str) -> list[dict[str, Any] | str]:
    """Decode the serialized MCP server list passed by the harness."""
    servers = json.loads(raw)
    if not isinstance(servers, list):
        raise ValueError("--mcp-servers-json must decode to a list")
    return servers


def build_mcp_section(servers: list[dict[str, Any] | str]) -> str:
    """Render MCP discovery text for the prompt without launching servers."""
    if not servers:
        return ""

    lines = [
        "",
        "",
        "MCP Servers:",
        "The following MCP servers are available for this task.",
    ]
    for server in servers:
        if isinstance(server, str):
            lines.append(f"- {server}")
            continue

        name = server.get("name", "unnamed")
        transport = server.get("transport", server.get("type", "stdio"))
        if transport == "stdio":
            args = " ".join(str(arg) for arg in server.get("args", []))
            command = " ".join(
                part for part in [str(server.get("command", "")), args] if part
            )
            lines.append(f"- {name}: stdio transport, command: {command}")
        else:
            lines.append(
                f"- {name}: {transport} transport, url: {server.get('url', '')}"
            )
    return "\n".join(lines) + "\n"


def parse_skill_frontmatter(content: str) -> dict[str, str] | None:
    """Extract name and description from simple key-value frontmatter."""
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None

    frontmatter: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip("\"'")

    if "name" not in frontmatter or "description" not in frontmatter:
        return None
    return {
        "name": frontmatter["name"],
        "description": frontmatter["description"],
    }


def build_skills_section(skills_dir: str | None) -> str:
    """Render one-level Agent Skills discovery as XML for the model prompt."""
    if not skills_dir:
        return ""

    root_path = Path(skills_dir)
    if not root_path.is_dir():
        return ""

    root = Element("available_skills")
    for skill_path in sorted(root_path.glob("*/SKILL.md")):
        frontmatter = parse_skill_frontmatter(skill_path.read_text())
        if not frontmatter:
            continue
        skill = SubElement(root, "skill")
        SubElement(skill, "name").text = frontmatter["name"]
        SubElement(skill, "description").text = frontmatter["description"]
        SubElement(skill, "location").text = str(skill_path)

    if not list(root):
        return ""

    indent(root, space="  ")
    return "\n" + tostring(root, encoding="unicode")


def build_augmented_instruction(
    instruction: str,
    mcp_servers: list[dict[str, Any] | str],
    skills_dir: str | None,
) -> str:
    """Append MCP and Agent Skills discovery sections to the task."""
    return (
        instruction + build_mcp_section(mcp_servers) + build_skills_section(skills_dir)
    )


def count_message_chars(messages: list[dict[str, str]]) -> int:
    """Count message content characters for summarization threshold checks."""
    return sum(len(message.get("content", "")) for message in messages)


def should_summarize(
    messages: list[dict[str, str]],
    threshold_chars: int,
    keep_messages: int,
) -> bool:
    """Decide whether history is large enough to trigger summarization.

    The turn count ignores system messages, while the character budget includes
    all message content.
    """
    non_system_count = len([m for m in messages if m["role"] != "system"])
    return (
        threshold_chars > 0
        and non_system_count > keep_messages
        and count_message_chars(messages) > threshold_chars
    )


def build_summary_prompt(original_instruction: str) -> str:
    """Ask the current agent to summarize work for a handoff."""
    return f"""You are about to hand off your work to another AI agent.
Please provide a comprehensive summary of what you have accomplished so far on this task:

Original Task: {original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. Major Actions Completed - List each significant command you executed and what you learned from it.
2. Important Information Learned - Summarize crucial findings, file locations, configurations, error messages, or system state discovered.
3. Challenging Problems Addressed - Include any significant issues you encountered and how you resolved them.
4. Current Status - Explain exactly where you are in the task completion process.

Be comprehensive and detailed. The next agent needs to understand everything that has happened so far in order to continue."""


def build_question_prompt(
    original_instruction: str,
    summary: str,
    current_screen: str,
) -> str:
    """Ask the next agent what missing context it needs before handoff."""
    return f"""You are picking up work from a previous AI agent on this task:

Original Task:
{original_instruction}

Summary from Previous Agent:
{summary}

Current Terminal Screen:
{current_screen}

Please begin by asking several questions, at least five and more if necessary, about the current state of the solution that are not answered in the summary from the prior agent. After you ask these questions you will be on your own, so ask everything you need to know."""


def build_handoff_prompt(answers: str) -> str:
    """Package handoff answers as the new working prompt."""
    return (
        "Here are the answers the other agent provided.\n\n"
        f"{answers}\n\n"
        "Continue working on this task from where the previous agent left off. "
        "You can no longer ask questions. Please follow the spec to interact with the terminal."
    )


def parse_xml_response(response: str) -> dict[str, Any]:
    """Parse the XML response variant into the same command payload shape."""
    start = response.find("<response>")
    end = response.find("</response>", start)
    if start == -1:
        return {
            "commands": [],
            "task_complete": False,
            "error": "No <response> tag found",
        }
    if end == -1:
        content = response[start + len("<response>") :].strip()
    else:
        content = response[start + len("<response>") : end].strip()

    missing = [
        tag
        for tag in ("analysis", "plan", "commands")
        if extract_tag(content, tag) is None
    ]
    if missing:
        return {
            "commands": [],
            "task_complete": False,
            "error": f"Missing XML sections: {', '.join(missing)}",
        }

    command_block = extract_tag(content, "commands") or ""
    commands = []
    matches = re.findall(
        r"<keystrokes([^>]*)>(.*?)</keystrokes>",
        command_block,
        flags=re.DOTALL,
    )
    for attrs, keystrokes in matches:
        duration = 1.0
        duration_match = re.search(r"duration\s*=\s*[\"']([^\"']*)[\"']", attrs)
        if duration_match:
            try:
                duration = float(duration_match.group(1))
            except ValueError:
                duration = 1.0
        commands.append({"keystrokes": keystrokes, "duration": duration})

    complete_value = extract_tag(content, "task_complete")
    complete = bool(
        complete_value and complete_value.strip().lower() in {"true", "1", "yes"}
    )
    return {"commands": commands, "task_complete": complete, "error": ""}


def limit_output(output: str, max_bytes: int) -> str:
    """Keep terminal output within byte limits while preserving head and tail."""
    encoded = output.encode()
    if len(encoded) <= max_bytes:
        return output
    half = max_bytes // 2
    first = encoded[:half].decode(errors="ignore")
    last = encoded[-half:].decode(errors="ignore")
    omitted = len(encoded) - len(first.encode()) - len(last.encode())
    return f"{first}\n[... output limited to {max_bytes} bytes; {omitted} bytes omitted ...]\n{last}"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI used by the sandbox-installed Terminus runner."""
    parser = argparse.ArgumentParser(
        description="Run a Terminus 2-style tmux agent loop."
    )
    parser.add_argument("--instruction-path", required=True)
    parser.add_argument("--template-dir", default="/opt/terminus_2/templates")
    parser.add_argument("--system-prompt-path", default="/terminus_2/system_prompt.txt")
    parser.add_argument("--parser-name", choices=["json", "xml"], default="json")
    parser.add_argument("--agent-workdir", default="/app")
    parser.add_argument("--log-path", default="/logs/agent/terminus_2.log")
    parser.add_argument("--mcp-servers-json", default="[]")
    parser.add_argument("--skills-dir")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--tmux-width", type=int, default=160)
    parser.add_argument("--tmux-height", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-output-bytes", type=int, default=10_000)
    parser.add_argument(
        "--disable-summarize", action="store_false", dest="enable_summarize"
    )
    parser.add_argument("--summarization-threshold-chars", type=int, default=120_000)
    parser.add_argument("--summarization-keep-messages", type=int, default=4)
    parser.set_defaults(enable_summarize=True)
    return parser


def main() -> int:
    """Parse CLI arguments and run the Terminus runner."""
    args = build_arg_parser().parse_args()
    try:
        return TerminusRunner(args).run()
    except Exception as e:
        print(f"terminus_2 runner failed: {e}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
