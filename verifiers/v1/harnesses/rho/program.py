# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "agent-client-protocol==0.11.0",
#     "openai==2.49.0",
#     "mcp==1.28.1",
#     "httpx==0.28.1",
#     "tenacity==9.1.4",
#     "jsonschema==4.25.1",
#     # Office-format libraries for run_code: without them models hand-roll OOXML
#     # zips and zlib-decode PDF streams instead of doing the task.
#     "openpyxl==3.1.5",
#     "pypdf==6.14.2",
#     "python-docx==1.2.0",
#     "python-pptx==1.0.2",
# ]
# ///
"""In-box driver for the rho harness: pi's four tools plus run_code.

Five native tools — read, write, edit, bash (pi semantics), and run_code. Everything
compositional lives inside run_code: the task's MCP tools are documented as files under
./tools/<app>/<verb> and pre-bound as `<app>.<verb>(...)` callables, `completion()` makes
one-shot sub-LLM calls, and `agent()` spawns subagents (config-gated, depth 1). Model code
executes in a persistent kernel subprocess reached over stdio NDJSON; the same channel is
the tool bridge, so the kernel holds capabilities, never credentials.

The program is an ACP agent: the harness's runner spawns it once per rollout, session/new
builds the driver and its MCP surface, and each session/prompt runs one loop segment on
the same conversation — kernel namespace, named agent() sessions, and the transcript all
persist across segments. Config rides RHO_* environment variables, popped on read so the
API secret never reaches the kernel's or bash's inherited environment; the final reply of
each segment goes back as an agent message chunk.

The transcript is append-only between compaction boundaries: assistant messages (reasoning
included) are re-sent complete, and the only rewriting event is Codex-style checkpoint
compaction — triggered by the token budget, a context overflow, or the model's own
`compact` tool call.

Two standing recovery stores: the live transcript (every root-loop message appended to
one greppable file as it happens; subagent transcripts land beside it under agents/) and
spill files (full payloads named by truncation footers, whose pointers ride inside the
transcript). `agent()` supports named persistent sessions — transcript, tool surface,
and kernel continue across calls; sessions survive checkpoints and later prompt
segments alike, because the process is the rollout.

Deliberate absences (known, may be wanted later, not built yet):
- Parallel tool calls: multiple calls in one turn execute sequentially, and the kernel
  bridge is a serial stdio channel — code-side `gather` over tools needs bridge
  multiplexing (frame ids), one coherent upgrade with interception-side concurrency.
- Background tools: no `background`/yield machinery (Codex's session+wait stack); the
  documented answer for long-running processes is bash: nohup, redirect to a file, read.
- Session compaction: a session that outgrows the model context is refused with a fact
  (start a fresh session), not compacted — subagent loops have no checkpoint machinery.
"""

import asyncio
import base64
import contextlib
import itertools
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from acp import PROTOCOL_VERSION, Agent, Client, run_agent, update_agent_message_text
from acp.schema import (
    AgentCapabilities,
    ImageContentBlock,
    InitializeResponse,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    TextContentBlock,
)
from openai import AsyncOpenAI
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

MAX_LINES = 2000
MAX_BYTES = 50 * 1024
MAX_LINE_CHARS = 2000
"""Three ceilings on every output path: line window, byte budget, per-line clamp.
Omitting any one leaves a file shape unhandled (one minified line inside the line
window can eat the whole byte budget)."""

SPILL_DIR = Path("/tmp/.rho")
"""Overflowing output is spilled here in full; the truncation footer names the path so
recovery is a read/grep, not a guess. Outside the workspace so spills never pollute
task deliverables."""

CELL_TIMEOUT = 30.0
"""run_code compute budget per cell, seconds. The clock pauses while bridged tool calls
are in flight, so tool latency never triggers it. Long pure compute is not a knob — it
is bash: write a script, run it under the tool with no timeout."""

KERNEL_KILL_GRACE = 5.0
"""After SIGINT, seconds to wait for the cell to unwind before escalating to SIGKILL."""

IMAGE_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"GIF8": "image/gif",
}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def image_mime(head: bytes) -> str | None:
    for magic, mime in IMAGE_MAGIC.items():
        if head.startswith(magic):
            return mime
    # RIFF alone matches WAV/AVI too; webp needs the form type at offset 8.
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return None


SPAWN_RESERVE_TURNS = 3
"""Refuse to delegate when fewer turns than this remain: a subagent can spend the whole
shared budget, but the parent must keep enough to act on the result. A reserve, not a
cap — the box imposes no limits of its own; the framework's turn budget is the bound."""

TUNNEL_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
TUNNEL_POOL = httpx.Limits(
    max_connections=16, max_keepalive_connections=16, keepalive_expiry=900.0
)
"""Everything the box talks to is the host reached back through the runtime's tunnel,
where opening a connection is the slow part (measured 3-16s, up to 55s). Pool for the
whole rollout so that cost is paid per rollout, not per call."""

MCP_CALL_ATTEMPTS = 6
TOOLS_DIR = "tools"
DISCOVERY_CATALOG = "tools_catalog"
RUNTIME_DIAGNOSTICS = "/tmp/.rho/runtime.json"
"""Box-side stats, inside the one harness-owned dir (mirrored in harness.py).
Per-segment and last-write-wins on resume — cumulative truth lives on the trace."""

OVERFLOW_PATTERNS = (
    "context length",
    "context_length",
    "prompt is too long",
    "exceeds the context window",
    "maximum context",
    "too many tokens",
    "input is too long",
)

# Checkpoint prompt shared verbatim with openai/codex (compact.rs), plus the kernel
# persistence note Codex lacks and pi's update-style wording for repeat compactions.
COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another "
    "LLM that will resume the task.\n"
    "\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n"
    "\n"
    "Be concise, structured, and focused on helping the next LLM seamlessly continue the work."
    "\n\n"
    "Note: the workspace and the Python namespace used by run_code stay as they are across "
    "this compaction — files you wrote and variables you defined are still there. Mention "
    "important file paths and variable names so the next LLM knows what's available.\n"
    "If the conversation already contains a previous checkpoint summary, update it: preserve "
    "all information that is still relevant and move completed items to done — do not drop "
    "carried context."
)
COMPACTION_FRAMING = (
    "Another language model started to solve this problem and produced a summary of its "
    "thinking process. It worked in this same environment: the workspace files and run_code "
    "variables it mentions are still present. Use this to build on the work that has already "
    "been done and avoid duplicating work. Here is the summary produced by the other language "
    "model, use the information in this summary to assist with your own analysis:"
)

TRANSCRIPT_PATH = SPILL_DIR / "transcript.txt"
"""The live transcript: every root-loop message is appended here as it happens, so the
model can grep its own history at any time — not only after a checkpoint. One stable
path per runtime, surviving compactions and resumed segments."""

# With the transcript file, the summary is licensed to be tight: raw data stays
# greppable, so the checkpoint carries decisions and state, not hedged payloads.
TRANSCRIPT_PROMPT_NOTE = (
    "\nThe full transcript you are summarizing will remain available to the next LLM in a "
    "file it can grep, so keep the summary tight and targeted — decisions, state, and next "
    "steps. Raw data and exact quotes can be recovered from the file, or from the spill "
    "files its truncation footers name."
)
TRANSCRIPT_FRAMING_NOTE = (
    "\n\nThe full transcript (including everything this summary replaced) is at {path} — "
    "grep it if a detail you need is missing from the summary. Work from evidence: "
    "the workspace is authoritative — inspect the current state of files before relying "
    "on the summary or the history."
)
TRANSCRIPT_SYSTEM_NOTE = (
    "Your full transcript so far is appended to {path} — read or grep it to recover "
    "earlier details, tool outputs, or exact quotes at any time."
)
SESSIONS_CHECKPOINT_CLAUSE = (
    " Named agent() sessions also persist across this compaction."
)


def render_history(messages: list[dict]) -> str:
    """Render messages as greppable role-tagged text (the transcript file format)."""
    blocks = []
    for m in messages:
        role = m.get("role", "?")
        parts = []
        if m.get("reasoning_content"):
            parts.append(f"(reasoning)\n{m['reasoning_content']}")
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                else:
                    parts.append(f"[{part.get('type', 'attachment')}]")
        elif content:
            parts.append(str(content))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            parts.append(f"(tool call) {fn.get('name')}({fn.get('arguments', '')})")
        blocks.append(f"[{role}]\n" + "\n".join(parts))
    return "\n\n".join(blocks) + "\n"


def write_diagnostics(phase: str, **values) -> None:
    SPILL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"phase": phase, "python": platform.python_version(), **values}
    Path(RUNTIME_DIAGNOSTICS).write_text(json.dumps(payload, sort_keys=True))


# --------------------------------------------------------------------------------------
# Output bounding: three ceilings, tail-kept, spill to a named path
# --------------------------------------------------------------------------------------


def _spill(text: str, label: str) -> str:
    # mkstemp, not a counter: the counter resets per process, and a resumed segment
    # (or a shared /tmp) would overwrite files an earlier transcript still names.
    SPILL_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=SPILL_DIR, prefix=f"{label}-", suffix=".txt")
    os.write(fd, text.encode())
    os.close(fd)
    return path


def cap_output(text: str, label: str) -> str:
    """Bound execution output (bash, run_code): keep the tail, spill the full text."""
    if not text.strip():
        return "(no output)"
    lines = text.splitlines()
    total = len(lines)
    # Walk the tail window backwards, clamping and accumulating until the byte
    # budget fills — one pass over at most MAX_LINES lines, however big the text.
    kept: list[str] = []
    size = 0
    clipped = False
    for line in reversed(lines[-MAX_LINES:]):
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + f" …[line clipped, {len(line)} chars total]"
            clipped = True
        if kept and size + len(line) + 1 > MAX_BYTES:
            break
        size += len(line) + 1
        kept.append(line)
    kept.reverse()
    if len(kept) == total and not clipped:
        return "\n".join(kept)
    path = _spill(text, label)
    start = total - len(kept) + 1
    return (
        "\n".join(kept)
        + f"\n[Showing lines {start}-{total} of {total}. Full output: {path} — read or grep it, or print less.]"
    )


# --------------------------------------------------------------------------------------
# Native tools: read / write / edit / bash (pi semantics)
# --------------------------------------------------------------------------------------

READ_LEDGER: set[str] = set()
"""Paths read this rollout; write consults it before overwriting an existing file."""


def _resolve(path: str, cwd: Path | None) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (cwd or Path.cwd()) / p


def run_read(path: str, offset=None, limit=None, cwd: Path | None = None):
    p = _resolve(path, cwd)
    if not p.exists():
        return f"{path} not found"
    if p.is_dir():
        entries = sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir())
        return (
            f"{path} is a directory ({len(entries)} entries) — use bash ls; first entries: "
            + ", ".join(entries[:20])
        )
    with p.open("rb") as f:
        head = f.read(12)
    mime = image_mime(head)
    if mime is not None:
        size = p.stat().st_size
        if size > MAX_IMAGE_BYTES:
            return (
                f"{path} is a {mime} of {size} bytes — over the {MAX_IMAGE_BYTES} byte "
                "image cap; downscale it first (bash or run_code)"
            )
        READ_LEDGER.add(str(p))
        encoded = base64.b64encode(p.read_bytes()).decode()
        return [
            {"type": "text", "text": f"{path} ({mime}, {size} bytes)"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            },
        ]

    try:
        offset = max(1, int(offset)) if offset is not None else 1
        limit = max(1, int(limit)) if limit is not None else MAX_LINES
    except (TypeError, ValueError):
        return "offset and limit must be integers"
    limit = min(limit, MAX_LINES)

    total = 0
    window: list[str] = []
    budget = MAX_BYTES
    clipped_mid_line = False
    more_follows = False
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            total = i
            if i < offset:
                continue
            if len(window) >= limit or budget <= 0:
                # The window is full: one extra line proves more follows, and the
                # scan stops — paging a huge file must not decode its whole tail.
                more_follows = True
                break
            line = line.rstrip("\n")
            if len(line) > MAX_LINE_CHARS:
                line = (
                    line[:MAX_LINE_CHARS]
                    + " …[line clipped; read the file to see the rest]"
                )
            if len(line) + 1 > budget:
                window.append(line[:budget])
                budget = 0
                clipped_mid_line = True
                continue
            budget -= len(line) + 1
            window.append(line)

    READ_LEDGER.add(str(p))
    if total == 0:
        return f"{path} is empty"
    if offset > total:
        return f"offset {offset} is past the end of {path} — {total} lines total"
    end = offset + len(window) - 1
    body = "\n".join(window)
    if more_follows or clipped_mid_line:
        # Resume ON the clipped line when the byte cap cut it short, else the next line.
        resume = end if clipped_mid_line else end + 1
        return (
            body
            + f"\n[Showing lines {offset}-{end}; more of the file follows. Use offset={resume} to continue.]"
        )
    return body


def run_write(path: str, content: str, cwd: Path | None = None) -> str:
    p = _resolve(path, cwd)
    if p.exists() and p.stat().st_size > 0 and str(p) not in READ_LEDGER:
        return (
            f"refusing to overwrite {path}: it exists and was never read this session — "
            "read it first, or write to a new path"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    READ_LEDGER.add(str(p))
    return f"Wrote {len(content.encode())} bytes to {path}"


def run_edit(path: str, edits, cwd: Path | None = None) -> str:
    if not isinstance(edits, list) or not edits:
        return "edits must be a non-empty array of {oldText, newText}"
    p = _resolve(path, cwd)
    if not p.exists():
        return f"{path} not found"
    try:
        # newline="" preserves the file's own line endings: a localized edit must not
        # silently rewrite CRLF to LF across every untouched line.
        with p.open("r", newline="") as f:
            original = f.read()
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"could not read {path}: {e}"

    spans: list[tuple[int, int, str]] = []
    for i, edit in enumerate(edits):
        old, new = edit.get("oldText"), edit.get("newText")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return (
                f"edits[{i}]: oldText must be a non-empty string and newText a string"
            )
        count = original.count(old)
        if count != 1:
            return f"edits[{i}]: oldText must appear exactly once in the original {path} (found {count})"
        start = original.index(old)
        spans.append((start, start + len(old), new))
    spans.sort()
    for (_, prev_end, _), (start, _, _) in itertools.pairwise(spans):
        if start < prev_end:
            return "edits overlap in the original file — merge them into one edit"

    result, cursor = [], 0
    for start, end, new in spans:
        result.append(original[cursor:start])
        result.append(new)
        cursor = end
    result.append(original[cursor:])
    try:
        with p.open("w", newline="") as f:
            f.write("".join(result))
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"could not write {path}: {e}"
    READ_LEDGER.add(str(p))
    return f"Applied {len(spans)} edit(s) to {path}"


def run_bash(command: str, timeout=None, cwd: Path | None = None) -> str:
    try:
        timeout = float(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        return "timeout must be a number of seconds"
    proc = subprocess.Popen(
        ["bash", "-c", command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        try:
            # Bounded: a descendant that escaped the process group can hold the
            # stdout pipe open forever; the tool must return regardless.
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.stdout.close()
            out = ""
        return cap_output(
            (out or "") + f"\n[command timed out after {timeout:g}s and was killed]",
            "bash",
        )
    text = out or "(no output)"
    if proc.returncode != 0:
        text += f"\n[exit code {proc.returncode}]"
    return cap_output(text, "bash")


# --------------------------------------------------------------------------------------
# The kernel: persistent Python subprocess, stdio NDJSON = cells + tool bridge
# --------------------------------------------------------------------------------------

RUNNER_SOURCE = r'''
import builtins, json, os, sys, tempfile, traceback

PROTO = os.fdopen(os.dup(1), "w", buffering=1)
DEVNULL = os.open(os.devnull, os.O_WRONLY)
os.dup2(DEVNULL, 1)
os.dup2(DEVNULL, 2)

NS = {"__name__": "__main__"}


def send(frame):
    PROTO.write(json.dumps(frame) + "\n")
    PROTO.flush()


def bridge(kind, payload):
    send({"call": {"kind": kind, **payload}})
    line = sys.stdin.readline()
    if not line:
        os._exit(0)
    frame = json.loads(line)
    if "result" in frame:
        return frame["result"]
    err = frame.get("error")
    if err is None:
        raise RuntimeError(f"unexpected bridge frame: {frame!r}")
    exc = getattr(builtins, err.get("type", ""), None)
    if isinstance(exc, type) and issubclass(exc, BaseException):
        raise exc(err.get("message", ""))
    raise RuntimeError(err.get("message", ""))


class App:
    """One service app: verbs resolve on attribute access, calls go over the bridge."""

    def __init__(self, name, verbs):
        self._name = name
        self._verbs = verbs  # verb -> param name list (empty for documented-only verbs)

    def __getattr__(self, verb):
        params = self._verbs.get(verb, [])

        def proxy(*args, **kwargs):
            if len(args) > len(params):
                raise TypeError(f"{self._name}.{verb}() takes at most {len(params)} positional arguments ({len(args)} given)")
            positional = dict(zip(params, args))
            collisions = set(positional) & set(kwargs)
            if collisions:
                raise TypeError(f"{self._name}.{verb}() got multiple values for {sorted(collisions)}")
            kwargs.update(positional)
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return bridge("mcp", {"app": self._name, "verb": verb, "args": kwargs})

        return proxy


def init(payload):
    apps = {name: App(name, verbs) for name, verbs in payload.get("apps", {}).items()}
    NS.update(apps)
    NS["json"] = json

    # `import <app>` resolves to the service proxy — measured rollouts hit this friction.
    real_import = builtins.__import__

    def import_app_or_python(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in apps:
            return apps[name]
        return real_import(name, globals, locals, fromlist, level)

    NS["__builtins__"] = dict(vars(builtins), __import__=import_app_or_python)

    def completion(prompt, schema=None, system=None):
        return bridge("completion", {"prompt": prompt, "schema": schema, "system": system})

    NS["completion"] = completion

    if payload.get("subagents"):
        def agent(prompt, schema=None, effort=None, tools=None, session=None):
            return bridge("agent", {"prompt": prompt, "schema": schema, "effort": effort,
                                    "tools": tools, "session": session})

        NS["agent"] = agent
    send({"initialized": True})


def run_cell(code):
    fd, cap_path = tempfile.mkstemp(prefix=".rho-cell-")
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    error = None
    try:
        exec(compile(code, "<cell>", "exec"), NS)
    except KeyboardInterrupt:
        error = "KeyboardInterrupt: cell interrupted"
    except BaseException as e:
        tb = e.__traceback__.tb_next  # drop the runner's own exec frame
        error = "".join(traceback.format_exception(type(e), e, tb))
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(DEVNULL, 1)
        os.dup2(DEVNULL, 2)
        os.close(fd)
    with open(cap_path, "r", errors="replace") as f:
        output = f.read()
    os.unlink(cap_path)
    send({"done": {"output": output, "error": error}})


def main():
    for line in sys.stdin:
        try:
            frame = json.loads(line)
            if "init" in frame:
                init(frame["init"])
            elif "code" in frame:
                run_cell(frame["code"])
        except KeyboardInterrupt:
            continue


try:
    main()
except BaseException:
    send({"fatal": traceback.format_exc()})
'''

KERNEL_ENV_KEEP = (
    "PATH",
    "HOME",
    "LANG",
    "TERM",
    "TMPDIR",
    "PYTHONPATH",
    "VIRTUAL_ENV",
)
# No UV_ prefix: UV_INDEX_<NAME>_PASSWORD and credential-bearing index URLs ride it.
KERNEL_ENV_PREFIXES = ("LC_", "XDG_")


def kernel_env() -> dict[str, str]:
    """Allowlisted env for the kernel: model code gets capabilities, not credentials."""
    return {
        k: v
        for k, v in os.environ.items()
        if k in KERNEL_ENV_KEEP or k.startswith(KERNEL_ENV_PREFIXES)
    }


_RUNNER_PATH: str | None = None


def _shared_runner_path() -> str:
    """One runner file per process — RUNNER_SOURCE is constant, every kernel execs it."""
    global _RUNNER_PATH
    if _RUNNER_PATH is None:
        fd, _RUNNER_PATH = tempfile.mkstemp(prefix=".rho-runner-", suffix=".py")
        os.write(fd, RUNNER_SOURCE.encode())
        os.close(fd)
    return _RUNNER_PATH


class Kernel:
    """Driver side of the kernel: lifecycle, cells, and the paused-clock timeout."""

    def __init__(self, init_payload: dict):
        self._init_payload = init_payload
        self._proc: asyncio.subprocess.Process | None = None
        self._death_note = ""

    async def _ensure(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            _shared_runner_path(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=kernel_env(),
        )
        self._send({"init": self._init_payload})
        frame = await self._read_frame(timeout=30.0)
        if not (frame and frame.get("initialized")):
            raise RuntimeError(f"kernel failed to initialize: {frame!r}")

    def _send(self, frame: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(frame) + "\n").encode())

    async def _read_frame(self, timeout: float | None) -> dict | None:
        assert self._proc is not None and self._proc.stdout is not None
        line = await asyncio.wait_for(self._proc.stdout.readline(), timeout)
        if not line:
            return None
        return json.loads(line)

    def _mark_dead(self, note: str) -> None:
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()
        self._proc = None
        self._death_note = note

    async def run_cell(self, code: str, handle_call) -> str:
        """Execute one cell; `handle_call` answers bridge frames. The compute clock
        pauses while a bridged call is in flight."""
        restart_note = ""
        if self._death_note:
            restart_note = f"[{self._death_note} — fresh kernel started: variables lost, files intact.]\n"
            self._death_note = ""
        await self._ensure()
        self._send({"code": code})
        budget = CELL_TIMEOUT
        while True:
            started = time.monotonic()
            try:
                frame = await self._read_frame(timeout=budget)
            except TimeoutError:
                return restart_note + await self._interrupt()
            budget -= time.monotonic() - started
            if frame is None:
                self._mark_dead("kernel died running your cell")
                return (
                    restart_note
                    + "[kernel died running this cell — fresh kernel next call: variables lost, files intact.]"
                )
            if "fatal" in frame:
                self._mark_dead("kernel crashed")
                return restart_note + cap_output(
                    f"[kernel crashed]\n{frame['fatal']}", "run_code"
                )
            if "call" in frame:
                # Bridge time is the tool's, not the cell's: clock paused.
                try:
                    result = await handle_call(frame["call"])
                    self._send({"result": result})
                except BridgeError as e:
                    self._send({"error": {"type": e.type, "message": str(e)}})
                except Exception as e:  # noqa: BLE001 - bridge failures raise inside the kernel
                    self._send(
                        {
                            "error": {
                                "type": "RuntimeError",
                                "message": f"{type(e).__name__}: {e}",
                            }
                        }
                    )
                continue
            if "done" in frame:
                done = frame["done"]
                text = done.get("output") or ""
                if done.get("error"):
                    text = (text + "\n" if text else "") + done["error"]
                return restart_note + cap_output(text or "(ran, no output)", "run_code")

    async def _interrupt(self) -> str:
        assert self._proc is not None
        with contextlib.suppress(ProcessLookupError):
            self._proc.send_signal(signal.SIGINT)
        try:
            deadline = time.monotonic() + KERNEL_KILL_GRACE
            while True:
                frame = await self._read_frame(
                    timeout=max(0.1, deadline - time.monotonic())
                )
                if frame is None:
                    break
                if "done" in frame:
                    done = frame["done"]
                    partial = done.get("output") or ""
                    return cap_output(
                        partial
                        + f"\n[cell exceeded the {CELL_TIMEOUT:g}s compute budget and was interrupted; "
                        "variables and earlier statements' effects persist. For long compute, write a "
                        "script and run it with bash.]",
                        "run_code",
                    )
        except TimeoutError:
            pass
        self._mark_dead("cell was stuck (uninterruptible) and the kernel was killed")
        return (
            f"[cell exceeded the {CELL_TIMEOUT:g}s compute budget and could not be interrupted — "
            "kernel killed; fresh kernel next call: variables lost, files intact.]"
        )

    async def close(self) -> None:
        self._mark_dead("")


class BridgeError(Exception):
    """A bridge failure delivered into the kernel as a typed exception."""

    def __init__(self, type_: str, message: str):
        super().__init__(message)
        self.type = type_


# --------------------------------------------------------------------------------------
# MCP: pooled clients, retried sessions, docs-on-disk, decoy-aware dispatch
# --------------------------------------------------------------------------------------


def mcp_clients(config: dict) -> dict[str, httpx.AsyncClient]:
    return {
        name: httpx.AsyncClient(
            follow_redirects=True,
            timeout=TUNNEL_TIMEOUT,
            limits=TUNNEL_POOL,
            headers=spec.get("headers") or None,
        )
        for name, spec in config.get("mcpServers", {}).items()
    }


def _only_cause(error: Exception) -> Exception:
    while len(group := getattr(error, "exceptions", ())) == 1:
        error = group[0]
    return error


@contextlib.asynccontextmanager
async def mcp_session(http_client: httpx.AsyncClient, url: str):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    stack = contextlib.AsyncExitStack()
    cancelled: BaseException | None = None
    try:
        read, write, *_ = await stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        yield session
    except asyncio.CancelledError as error:
        cancelled = error
        raise
    finally:
        try:
            await stack.aclose()
        except Exception as error:  # noqa: BLE001 - teardown noise must not lose the result
            # The transport's real failure surfaces here as the task group unwinds;
            # when the body was cancelled it IS the story (and becomes retryable).
            if cancelled is not None:
                raise _only_cause(error) from cancelled
            # Teardown noise on a completed call must not lose the result.


async def with_retry(call):
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MCP_CALL_ATTEMPTS),
        wait=wait_exponential_jitter(initial=0.5, max=30),
        reraise=True,
    ):
        with attempt:
            return await call()


async def connect_mcp(config: dict, clients: dict):
    """Enumerate tools across all servers concurrently (startup pays the slowest
    tunnel, not the sum); return (dispatch {(app, verb) -> (server, raw, params)},
    docs, catalog ref)."""
    dispatch: dict[tuple[str, str], tuple[str, str, list[str]]] = {}
    docs: dict = {}
    catalog_server: tuple[str, str] | None = None
    servers = list(config.get("mcpServers", {}).items())

    async def list_tools(name: str, spec: dict):
        async with mcp_session(clients[name], spec["url"]) as session:
            return (await session.list_tools()).tools

    listed = await asyncio.gather(
        *(
            with_retry(lambda name=name, spec=spec: list_tools(name, spec))
            for name, spec in servers
        )
    )
    for (name, _spec), tools in zip(servers, listed):
        for tool in tools:
            if tool.name == DISCOVERY_CATALOG:
                # The box's own channel: read once at startup, never bound or documented.
                catalog_server = (name, tool.name)
                continue
            # Wire contract with task authors: tools are named <app>_<verb> (first
            # underscore splits); a verb-less name lands in the shared `misc` app.
            bare = (
                tool.name.split("_", 1)[1]
                if tool.name.startswith("tools_")
                else tool.name
            )
            app, _, verb = bare.partition("_")
            if not verb:
                app, verb = "misc", bare
            if (app, verb) in dispatch:
                other = dispatch[(app, verb)][0]
                raise ValueError(
                    f"tool name collision: {app}.{verb} is served by both {other!r} and {name!r}"
                )
            props = (tool.inputSchema or {}).get("properties", {})
            required = set((tool.inputSchema or {}).get("required", []))
            dispatch[(app, verb)] = (name, tool.name, list(props))
            params = ", ".join(p + ("" if p in required else "=...") for p in props)
            lines = [f"{app}.{verb}({params})", ""]
            if tool.description:
                lines.append(tool.description.strip())
            for p, s in props.items():
                lines.append(
                    f"  {p}: {s.get('type', 'any')}{' (required)' if p in required else ''}"
                    f"{' — ' + s['description'] if s.get('description') else ''}"
                )
            docs[(app, verb)] = "\n".join(lines) + "\n"
    return dispatch, docs, catalog_server


def mcp_content(result):
    parts = []
    for block in result.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts) if parts else str(result.content)


async def call_mcp_raw(servers, clients, server_name, raw_name, arguments):
    async def call():
        async with mcp_session(
            clients[server_name], servers[server_name]["url"]
        ) as session:
            return await session.call_tool(raw_name, arguments)

    result = await with_retry(call)
    text = mcp_content(result)
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001 - non-JSON results pass through as text
        return text


def _safe_component(name: str) -> str:
    """Doc-tree names come from the task's MCP server; keep them single path components.
    README is reserved for the tree's own summaries at both levels."""
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or name.startswith(".")
    ):
        raise ValueError(f"unsafe tool doc path component: {name!r}")
    if name.upper() == "README":
        raise ValueError(
            "app/verb name 'README' is reserved for the doc tree's summaries"
        )
    return name


def _fresh_tools_root(box: Path) -> Path:
    """Recreate the docs tree from nothing each time it is written. The tree is
    harness-owned; on a resumed segment the previous segment's model code may have
    replaced parts of it (a symlink at tools/<app> would route writes outside the
    box), so nothing pre-existing is trusted or followed."""
    root = box / TOOLS_DIR
    if root.is_symlink():
        root.unlink()
    elif root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def write_tool_docs(box: Path, docs: dict) -> None:
    root = _fresh_tools_root(box)
    for (app, verb), text in docs.items():
        d = root / _safe_component(app)
        d.mkdir(parents=True, exist_ok=True)
        (d / _safe_component(verb)).write_text(text)
    if docs:
        app, verb = next(iter(docs))
        (root / "README").write_text(
            "Each file under ./tools/<app>/<verb> documents one tool. The <app> names are "
            "pre-bound globals in run_code — call them directly; nothing to import.\n"
            "Worked example:\n"
            f"  bash:     cat tools/{app}/{verb}            # read the doc\n"
            f"  run_code: result = {app}.{verb}(...)        # call the pre-bound global\n"
            "  run_code: print(result)\n"
        )


def write_catalog_docs(box: Path, catalog: dict) -> None:
    """The discovery mode's doc tree: the whole documented surface, served or not."""
    if not catalog:
        return
    root = _fresh_tools_root(box)
    for app, entry in sorted(catalog.items()):
        directory = root / _safe_component(app)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "README").write_text(f"{app} — {entry['summary']}\n")
        for verb, doc in sorted(entry["verbs"].items()):
            (directory / _safe_component(verb)).write_text(doc)
    (root / "README").write_text(
        f"{len(catalog)} apps are documented here, one directory each. ./tools/<app>/README says\n"
        "what an app is, ./tools/<app>/<verb> documents one of its tools. The <app> names are\n"
        "pre-bound globals in run_code — call them directly; nothing to import.\n"
        "\n"
        "This deployment's catalog is documented in full; the workspace's integrations are not\n"
        "all provisioned. Calling an app this workspace does not have fails, and the work has to\n"
        "go through one it does.\n"
        "\n"
        "Worked example:\n"
        "  bash:     ls tools/                     # the apps this deployment documents\n"
        "  bash:     grep -ril refund tools/       # the docs that mention your work\n"
        "  bash:     cat tools/<app>/<verb>        # read one tool's doc\n"
        "  run_code: result = <app>.<verb>(...)    # call the pre-bound global\n"
        "  run_code: print(result)\n"
    )


# --------------------------------------------------------------------------------------
# Tool schemas (wire surface): the native tools plus compact, conditionally advertised
# --------------------------------------------------------------------------------------


def build_tool_schemas(tools: list[str]) -> list[dict]:
    def fn(name, description, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    schemas = {
        "read": fn(
            "read",
            "Read a file. Head-truncated with a resume offset; images are returned visually.",
            {
                "path": {"type": "string"},
                "offset": {"type": "integer", "description": "1-indexed first line"},
                "limit": {"type": "integer", "description": "max lines"},
            },
            ["path"],
        ),
        "write": fn(
            "write",
            "Write a file (creates parent directories; overwrites only files already read this session).",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        "edit": fn(
            "edit",
            "Edit a file by exact string replacement. Each oldText must appear exactly once "
            "in the original file; all edits are matched against the original and applied together.",
            {
                "path": {"type": "string"},
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            ["path", "edits"],
        ),
        "bash": fn(
            "bash",
            "Run a shell command in the workspace. No default timeout.",
            {
                "command": {"type": "string"},
                "timeout": {
                    "type": "number",
                    "description": "seconds (optional, no default)",
                },
            },
            ["command"],
        ),
        "run_code": fn(
            "run_code",
            "Run Python in a persistent kernel. Variables persist across calls and survive context "
            f"checkpoints. Up to {CELL_TIMEOUT:g}s of compute per cell (time inside tool calls does "
            "not count); for longer compute write a script and run it with bash. Use print(...) to "
            "see output. A cell that raises has already applied its earlier statements' effects — "
            "resume from the failure, don't re-run the whole cell.",
            {"code": {"type": "string"}},
            ["code"],
        ),
        "compact": fn(
            "compact",
            "Checkpoint your context before the next turn: the conversation is replaced by a "
            "handoff summary you write. Workspace files and run_code variables survive.",
            {},
            [],
        ),
    }
    return [schemas[name] for name in tools if name in schemas]


# --------------------------------------------------------------------------------------
# The model loop: shared turn budget, compaction, overflow recovery
# --------------------------------------------------------------------------------------


class TurnBudget:
    """One counter for every model call this rollout makes — work turns, compactions,
    completions, subagent turns — mirroring the framework's single per-trace cap."""

    def __init__(self, max_turns: int | None):
        self.max_turns = max_turns  # None = no cap set by the framework: unbounded
        self.spent = 0

    def take(self) -> bool:
        if self.max_turns is not None and self.spent >= self.max_turns:
            return False
        self.spent += 1
        return True

    @property
    def remaining(self) -> int | None:
        return None if self.max_turns is None else self.max_turns - self.spent


def is_overflow_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(pattern in text for pattern in OVERFLOW_PATTERNS)


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def validate_schema(value, schema) -> str | None:
    """Best-effort JSON Schema validation; returns an error message or None."""
    try:
        import jsonschema

        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as e:
        return e.message
    except Exception:  # noqa: BLE001 - validation is best-effort, never fatal
        return None
    return None


def schema_suffix(lead: str, schema) -> str:
    return (
        f"\n\n{lead} only valid JSON matching this schema (no prose, no code fences):\n"
        + json.dumps(schema)
    )


def parse_reply(text: str, schema) -> tuple[object, str | None]:
    """Parse a schema-bound model reply; returns (value, None) or (None, problem)."""
    try:
        value = json.loads(strip_fences(text))
    except Exception:  # noqa: BLE001 - malformed JSON is retried with feedback
        return None, "the reply was not valid JSON"
    problem = validate_schema(value, schema)
    return (value, None) if problem is None else (None, problem)


class Driver:
    """Owns the client, kernel, MCP dispatch, and both loops (main + subagents)."""

    def __init__(self, config: "Config"):
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            http_client=httpx.AsyncClient(timeout=TUNNEL_TIMEOUT, limits=TUNNEL_POOL),
        )
        self.budget = TurnBudget(config.max_turns)
        self.box = Path.cwd()
        self.tools: list[str] = list(config.tools)
        self.kernel: Kernel | None = None
        self.apps_payload: dict = {}
        self.dispatch: dict = {}
        self.servers: dict = {}
        self.clients: dict = {}
        self.documented_apps: set[str] = set()
        self.tool_discovery: dict[str, int] = {}
        self.subagents_spawned = 0
        self.completions = 0
        self.transcript_pos = 0
        """Messages of the root loop already appended to the live transcript file."""
        self.sessions: dict[str, dict] = {}
        """Named persistent subagents: session name -> {"messages", "tools"}. A
        continued session appends the new prompt and keeps its context — the caller
        pays for the delta, not a re-briefing — and extends the same trace branch."""
        self.compactions = 0
        self.compaction_pending = False
        self.just_compacted = False
        self.auto_compact_disabled = False
        self.last_prompt_tokens = 0
        self.peak_prompt_tokens = 0

    # --- setup ------------------------------------------------------------------

    @property
    def served_apps(self) -> set[str]:
        return {app for app, _ in self.dispatch}

    async def setup(self, mcp_config: dict) -> None:
        self.servers = mcp_config.get("mcpServers", {})
        self.clients = mcp_clients(mcp_config)
        self.dispatch, docs, catalog_ref = await connect_mcp(mcp_config, self.clients)
        apps_payload = {}
        for (app, verb), (_, _, params) in self.dispatch.items():
            apps_payload.setdefault(app, {})[verb] = params
        if catalog_ref is not None:
            server, raw = catalog_ref
            catalog = await call_mcp_raw(self.servers, self.clients, server, raw, {})
            write_catalog_docs(self.box, catalog)
            self.documented_apps = set(catalog)
            self.tool_discovery.update(
                decoy_calls=0,
                surface_apps=len(catalog),
                surface_tools=sum(len(e["verbs"]) for e in catalog.values()),
                served_apps=len(self.served_apps),
            )
            for app in catalog:
                apps_payload.setdefault(app, {})
        else:
            write_tool_docs(self.box, docs)
            self.documented_apps = set(self.served_apps)
        reserved = {"json", "completion", "agent"} & set(apps_payload)
        if reserved:
            raise ValueError(
                f"app names collide with run_code globals: {sorted(reserved)}"
            )
        self.apps_payload = apps_payload
        if "run_code" in self.tools:
            self.kernel = Kernel(
                {"apps": apps_payload, "subagents": self.config.subagents}
            )

    # --- bridge handlers ----------------------------------------------------------

    async def handle_call(self, call: dict, depth: int):
        kind = call.get("kind")
        if kind == "mcp":
            return await self.do_mcp(
                call.get("app", ""), call.get("verb", ""), call.get("args") or {}
            )
        if kind == "completion":
            return await self.do_completion(
                call.get("prompt", ""), call.get("schema"), call.get("system")
            )
        if kind == "agent":
            if not self.config.subagents:
                raise BridgeError("NameError", "agent() is not enabled in this session")
            if depth > 0:
                raise BridgeError(
                    "RuntimeError", "subagents cannot spawn subagents (depth 1)"
                )
            return await self.do_agent(
                call.get("prompt", ""),
                call.get("schema"),
                call.get("effort"),
                call.get("tools"),
                call.get("session"),
            )
        raise BridgeError("RuntimeError", f"unknown bridge call {kind!r}")

    async def do_mcp(self, app: str, verb: str, arguments: dict):
        entry = self.dispatch.get((app, verb))
        if entry is None:
            if app in self.served_apps:
                raise BridgeError(
                    "AttributeError",
                    f"{app} has no {verb!r} tool — its tools are documented in ./tools/{app}/",
                )
            if app in self.documented_apps:
                self.tool_discovery["decoy_calls"] = (
                    self.tool_discovery.get("decoy_calls", 0) + 1
                )
                raise BridgeError(
                    "NameError",
                    f"no such app in this workspace: {app!r} — it is documented, but this workspace "
                    "has no such integration; the work has to go through an app that is connected",
                )
            raise BridgeError(
                "NameError",
                f"no tool named {app}.{verb!r} — explore ./tools/<app>/ and call <app>.<verb>(...)",
            )
        server, raw, _ = entry
        return await call_mcp_raw(self.servers, self.clients, server, raw, arguments)

    async def do_completion(self, prompt: str, schema, system):
        self.completions += 1
        if not self.budget.take():
            raise BridgeError(
                "RuntimeError", "turn budget exhausted — no model calls left"
            )
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        body = str(prompt) + (
            schema_suffix("Respond with", schema) if schema is not None else ""
        )
        messages.append({"role": "user", "content": body})
        problem = "no attempts left"
        for attempt in range(2):
            if attempt and not self.budget.take():
                break
            completion = await self.client.chat.completions.create(
                model=self.config.model, messages=messages, **self._effort_kwargs()
            )
            text = completion.choices[0].message.content or ""
            if schema is None:
                return text
            value, problem = parse_reply(text, schema)
            if problem is None:
                return value
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": f"{problem}. Reply with only the corrected JSON.",
                }
            )
        raise BridgeError(
            "ValueError",
            f"completion did not return JSON matching the schema: {problem}",
        )

    async def do_agent(self, prompt: str, schema, effort, tools, session=None):
        remaining = self.budget.remaining
        if remaining is not None and remaining < SPAWN_RESERVE_TURNS:
            raise BridgeError(
                "RuntimeError",
                f"not enough turn budget left to delegate ({remaining} turns remain) — "
                "finish the task directly",
            )
        body = str(prompt) + (
            schema_suffix("Your final message must be", schema)
            if schema is not None
            else ""
        )
        if session and session in self.sessions:
            # Continue the named session: same transcript, tool surface, and kernel —
            # the follow-up rides the context already paid for. The surface is locked
            # at creation (the stored system prompt was built from it).
            stored = self.sessions[session]
            messages, allowed, kernel = (
                stored["messages"],
                stored["tools"],
                stored["kernel"],
            )
            if tools is not None and sorted(tools) != sorted(allowed):
                raise BridgeError(
                    "ValueError",
                    f"session {session!r} already has tool surface {allowed}; "
                    "tools cannot change on continuation",
                )
            messages.append({"role": "user", "content": body})
        else:
            self.subagents_spawned += 1
            allowed = [t for t in self.tools if t != "compact"]
            if tools is not None:
                unknown = [t for t in tools if t not in allowed]
                if unknown:
                    raise BridgeError(
                        "ValueError",
                        f"unknown subagent tools: {unknown} (available: {allowed})",
                    )
                allowed = [t for t in allowed if t in tools]
            # A subagent gets its own kernel: agent() is only reachable from inside a
            # parent cell, whose kernel is blocked in bridge() — routing sub-cells to
            # it would corrupt the frame stream. Blank context means blank namespace.
            kernel = (
                Kernel({"apps": self.apps_payload, "subagents": False})
                if "run_code" in allowed
                else None
            )
            messages = [
                {"role": "system", "content": self.subagent_system_prompt(allowed)},
                {"role": "user", "content": body},
            ]
            if session:
                self.sessions[session] = {
                    "messages": messages,
                    "tools": allowed,
                    "kernel": kernel,
                }
        try:
            try:
                final = await self.run_loop(
                    messages,
                    tools=allowed,
                    depth=1,
                    # No per-spawn ceiling: allocation is the policy's call; the shared budget
                    # binds, minus the reserve that keeps the parent able to act.
                    max_turns=None
                    if self.budget.remaining is None
                    else self.budget.remaining - (SPAWN_RESERVE_TURNS - 1),
                    effort=effort,
                    kernel=kernel,
                )
            except Exception as error:
                # Subagent loops have no compaction by design: an outgrown context is
                # a fact with a recovery path, not a raw provider error in the cell.
                if is_overflow_error(error):
                    who = f"session {session!r}" if session else "the subagent"
                    raise BridgeError(
                        "RuntimeError",
                        f"{who} has outgrown the model context — start a new session "
                        "with a fresh, self-contained brief",
                    ) from error
                raise
            if final is None:
                return None
            if schema is None:
                return final
            problem = None
            for _ in range(2):
                value, problem = parse_reply(final, schema)
                if problem is None:
                    return value
                # Correction retries honor the same reserve as the spawn: they must
                # not drain the turns kept for the parent to act on the result.
                if (
                    self.budget.remaining is not None
                    and self.budget.remaining < SPAWN_RESERVE_TURNS
                ):
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": f"{problem}. Reply with only the corrected JSON.",
                    }
                )
                final = await self.run_loop(
                    messages,
                    tools=[],
                    depth=1,
                    max_turns=1,
                    effort=effort,
                    kernel=kernel,
                )
                if final is None:
                    break
            if final is not None:
                value, problem = parse_reply(final, schema)
                if problem is None:
                    return value
            raise BridgeError(
                "ValueError",
                f"subagent did not return JSON matching the schema: {problem}",
            )
        finally:
            self.save_agent_transcript(session, messages)
            if kernel is not None and not session:
                await kernel.close()

    def save_agent_transcript(self, session: str | None, messages: list[dict]) -> None:
        """Persist a subagent's full conversation under the agents/ recovery store.

        One mechanism, three affordances: subagent work is recoverable (the one
        otherwise-lossy channel), `ls` of the directory lists sessions, and a session
        name survives compaction even when the summary drops it."""
        name = session or f"anon-{self.subagents_spawned}"
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip(".") or "agent"
        with contextlib.suppress(Exception):
            agents_dir = SPILL_DIR / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            (agents_dir / f"{safe}.txt").write_text(render_history(messages))

    # --- native tool dispatch -------------------------------------------------------

    async def execute_tool(
        self, name: str, arguments: dict, depth: int, kernel: Kernel | None = None
    ):
        if name == "read":
            return await asyncio.to_thread(
                run_read,
                arguments.get("path", ""),
                arguments.get("offset"),
                arguments.get("limit"),
                self.box,
            )
        if name == "write":
            return await asyncio.to_thread(
                run_write,
                arguments.get("path", ""),
                arguments.get("content", ""),
                self.box,
            )
        if name == "edit":
            return await asyncio.to_thread(
                run_edit, arguments.get("path", ""), arguments.get("edits"), self.box
            )
        if name == "bash":
            return await asyncio.to_thread(
                run_bash,
                arguments.get("command", ""),
                arguments.get("timeout"),
                self.box,
            )
        if name == "run_code":
            kernel = kernel if kernel is not None else self.kernel
            assert kernel is not None
            return await kernel.run_cell(
                arguments.get("code", ""),
                lambda call: self.handle_call(call, depth=depth),
            )
        if name == "compact" and depth == 0:
            self.compaction_pending = True
            return "[context checkpoint scheduled before your next turn]"
        return f"unknown tool {name} — available: {', '.join(self.tools)}"

    # --- prompts --------------------------------------------------------------------

    def system_prompt_base(self, tools: list[str], advertise_agent: bool) -> str:
        parts = ["You are an agent operating in a minimal harness."]
        lines = []
        if "read" in tools:
            lines.append("`read` reads files (images render visually).")
        if "write" in tools or "edit" in tools:
            lines.append(
                "`write` creates files; `edit` does exact once-only string replacement."
                if "edit" in tools
                else "`write` creates files."
            )
        if "bash" in tools:
            lines.append("`bash` runs shell commands (no default timeout).")
        if "run_code" in tools:
            lines.append(
                "`run_code` runs Python in a persistent kernel: variables survive across calls "
                "and context checkpoints; never re-import or re-define what already exists. "
                "Chain tool results inside one cell only when their format is known — otherwise "
                "print first and continue next turn."
            )
            if self.served_apps or self.documented_apps:
                lines.append(
                    "Service tools are documented as files under ./tools/<app>/<verb> (explore with "
                    "bash) and pre-bound in run_code as <app>.<verb>(...) returning Python objects."
                )
            lines.append(
                "`completion(prompt, schema=None, system=None)` makes one standalone model call from code."
            )
            if advertise_agent:
                lines.append(
                    "`agent(prompt, schema=None, effort=None, tools=None, session=None)` runs a "
                    "subagent with a blank context and returns its final message (parsed when "
                    "schema is given); give it complete, self-contained instructions; its full "
                    "transcript is saved under /tmp/.rho/agents/. Pass a "
                    "session name to make it persistent: later calls with the same name continue "
                    "that subagent's conversation instead of starting over."
                )
        if "compact" in tools:
            lines.append(
                "`compact` checkpoints your context: call it when the transcript holds spent "
                "exploration you no longer need verbatim."
            )
        parts.append(" ".join(lines))
        return "\n\n".join(p for p in parts if p)

    def subagent_system_prompt(self, allowed: list[str]) -> str:
        # Built from the seat's actual surface: no advertised tools it can't call,
        # and never agent() — depth 1 rejects it.
        base = self.system_prompt_base(allowed, advertise_agent=False)
        return base + (
            "\n\nYou are a subagent. Work the task to completion; your final message is returned "
            "verbatim to the caller as data, not prose for a human."
        )

    # --- compaction -------------------------------------------------------------------

    def append_transcript(self, messages: list[dict]) -> None:
        """Append the root loop's not-yet-persisted messages to the live transcript."""
        new = messages[self.transcript_pos :]
        if not new:
            return
        SPILL_DIR.mkdir(parents=True, exist_ok=True)
        with TRANSCRIPT_PATH.open("a") as f:
            f.write(render_history(new) + "\n")
        self.transcript_pos = len(messages)

    async def checkpoint(self, messages: list[dict], tool_schemas: list[dict]) -> None:
        keep = 2  # the protected prefix: the system message + the opening user prompt
        # Normally a no-op — per-turn appends already persisted everything; kept as
        # a safety net so the file's completeness never depends on call-site order.
        self.append_transcript(messages)
        request = [
            *messages,
            {
                "role": "user",
                "content": COMPACTION_PROMPT
                + TRANSCRIPT_PROMPT_NOTE
                + (SESSIONS_CHECKPOINT_CLAUSE if self.sessions else ""),
            },
        ]
        while True:
            try:
                completion = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=request,
                    tools=tool_schemas or None,
                    tool_choice="none" if tool_schemas else None,
                    **self._effort_kwargs(),
                )
                break
            except Exception as error:
                # A compaction request that itself overflows trims oldest non-protected
                # messages and retries until it fits (Codex's trim loop). Drop one
                # message plus any contiguous tool results, so an assistant turn with
                # parallel tool calls never leaves orphaned tool messages behind.
                if is_overflow_error(error) and len(request) > keep + 2:
                    del request[keep]
                    while (
                        keep < len(request) - 1 and request[keep].get("role") == "tool"
                    ):
                        del request[keep]
                    continue
                raise
        summary = completion.choices[0].message.content or "(no summary available)"
        usage = getattr(completion, "usage", None)
        if usage and usage.prompt_tokens:
            self.peak_prompt_tokens = max(self.peak_prompt_tokens, usage.prompt_tokens)
        framing = (
            COMPACTION_FRAMING
            + "\n\n"
            + summary
            + TRANSCRIPT_FRAMING_NOTE.format(path=TRANSCRIPT_PATH)
        )
        messages[:] = [*messages[:keep], {"role": "user", "content": framing}]
        # The rebuilt prefix is already on disk; only the framing+summary is new.
        self.transcript_pos = keep
        self.append_transcript(messages)
        self.last_prompt_tokens = 0
        self.compactions += 1
        self.compaction_pending = False
        self.just_compacted = True

    def should_checkpoint(self) -> bool:
        # An explicit `compact` call is always honored; the automatic trigger stops once
        # a compaction proved ineffective (see the guard in `run_loop`) — no count cap,
        # matching Codex, because the turn budget is the real bound.
        if self.compaction_pending:
            return True
        return bool(
            not self.auto_compact_disabled
            and self.config.context_budget_tokens
            and self.last_prompt_tokens >= self.config.context_budget_tokens
        )

    # --- the loop ----------------------------------------------------------------------

    def _effort_kwargs(self, effort: str | None = None) -> dict:
        value = effort if effort is not None else self.config.effort
        return {"reasoning_effort": value} if value else {}

    def _notices(self) -> str:
        """Root-loop notices, appended to the turn's last tool result — the observation
        channel, never the system prompt, so every request extends the one before."""
        notices = []
        if "compact" in self.tools and self.last_prompt_tokens:
            notices.append(
                f"[harness] Context: ~{self.last_prompt_tokens} prompt tokens "
                f"(checkpoint threshold {self.config.context_budget_tokens})."
            )
        return ("\n\n" + "\n".join(notices)) if notices else ""

    async def run_loop(
        self,
        messages: list[dict],
        tools: list[str],
        depth: int,
        max_turns: int | None,
        effort: str | None = None,
        kernel: Kernel | None = None,
    ) -> str | None:
        """Drive one agent loop. Returns the final assistant text (None if none)."""
        tool_schemas = build_tool_schemas(tools)
        local_turns = 0
        final: str | None = None
        overflow_retried = False
        while max_turns is None or local_turns < max_turns:
            # A checkpoint is a model call on the same budget: only spend one while the
            # work turn it protects still fits under the cap.
            if (
                depth == 0
                and self.should_checkpoint()
                and (self.budget.remaining is None or self.budget.remaining > 1)
            ):
                self.budget.take()  # cannot fail: the guard ensured remaining > 1
                await self.checkpoint(messages, tool_schemas)
            if not self.budget.take():
                break
            try:
                completion = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=tool_schemas or None,
                    **self._effort_kwargs(effort),
                )
            except Exception as error:
                if depth == 0 and is_overflow_error(error) and not overflow_retried:
                    if self.budget.remaining is not None and self.budget.remaining <= 1:
                        # No budget for the checkpoint the retry needs: end cleanly
                        # with what we have instead of re-sending the same request.
                        return final
                    # A work turn rejected for context length forces a compaction and
                    # retries exactly once. The refused call cost nothing — refund it.
                    # (Assumes the interception server's ledger also skipped the refused
                    # call; if that ever drifts, the box just walks into the refused
                    # call the mirror was meant to avoid — degraded, not wrong.)
                    self.budget.spent -= 1
                    overflow_retried = True
                    self.compaction_pending = True
                    continue
                raise
            overflow_retried = False
            local_turns += 1
            usage = getattr(completion, "usage", None)
            if usage and usage.prompt_tokens:
                self.peak_prompt_tokens = max(
                    self.peak_prompt_tokens, usage.prompt_tokens
                )
            if usage and usage.prompt_tokens and depth == 0:
                self.last_prompt_tokens = usage.prompt_tokens
                if self.just_compacted:
                    # The first work turn after a checkpoint is the effectiveness probe:
                    # still over the threshold means the protected prefix itself is too
                    # big — further automatic compaction would loop without shrinking.
                    self.just_compacted = False
                    if (
                        self.config.context_budget_tokens
                        and usage.prompt_tokens >= self.config.context_budget_tokens
                    ):
                        self.auto_compact_disabled = True
            msg = completion.choices[0].message
            # Append complete — reasoning included. The transcript is append-only.
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                # Only a tool-free reply is a final answer; preparatory text alongside
                # tool calls must not pass as a truncated subagent's result.
                if msg.content:
                    final = msg.content
                break
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise TypeError("arguments must be a JSON object")
                except Exception as e:  # noqa: BLE001 - malformed calls become retryable tool errors
                    out = f"invalid tool arguments ({e}) — send a JSON object matching the tool's schema"
                else:
                    try:
                        out = await self.execute_tool(
                            tc.function.name, arguments, depth, kernel
                        )
                    except BridgeError as e:
                        out = f"{tc.function.name} failed: {e}"
                    except Exception as e:  # noqa: BLE001 - tool faults are observations, not rollout failures
                        out = f"{tc.function.name} failed: {type(e).__name__}: {e}"
                content = out if isinstance(out, list) else str(out)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": content}
                )
            if depth == 0:
                if isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += self._notices()
                self.append_transcript(messages)
                self.write_stats("loop")
        if depth == 0:
            self.append_transcript(messages)  # the final, tool-free assistant reply
        return final

    def write_stats(self, phase: str) -> None:
        transcript_bytes = 0
        with contextlib.suppress(OSError):
            transcript_bytes = TRANSCRIPT_PATH.stat().st_size
        write_diagnostics(
            phase,
            turns=self.budget.spent,
            compactions=self.compactions,
            auto_compact_disabled=self.auto_compact_disabled,
            peak_prompt_tokens=self.peak_prompt_tokens,
            subagents=self.subagents_spawned,
            completions=self.completions,
            sessions=sorted(self.sessions),
            transcript_bytes=transcript_bytes,
            **(
                {"tool_discovery": dict(self.tool_discovery)}
                if self.tool_discovery
                else {}
            ),
        )

    async def close(self) -> None:
        if self.kernel is not None:
            await self.kernel.close()
        for stored in self.sessions.values():
            if stored.get("kernel") is not None:
                await stored["kernel"].close()
        for client in self.clients.values():
            await client.aclose()


# --------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------


@dataclass
class Config:
    """The RHO_* environment contract with the harness (ACPConfig.env reaches the
    runner, which spawns this program with an inherited environment)."""

    base_url: str
    api_key: str
    model: str
    system_prompt: str
    effort: str
    tools: list[str]
    subagents: bool
    context_budget_tokens: int
    max_turns: int | None

    @classmethod
    def from_env(cls) -> "Config":
        env = os.environ
        system_prompt = ""
        if path := env.pop("RHO_SYSTEM_PROMPT_FILE", ""):
            # A file, not an env value: task system prompts can exceed env limits.
            file = Path(path)
            system_prompt = file.read_text()
            file.unlink(missing_ok=True)
        # Defaults below are manual-run conveniences; the harness always sets these.
        raw_tools = env.pop("RHO_TOOLS", "read,write,edit,bash,run_code")
        max_turns = env.pop("RHO_MAX_TURNS", "")
        return cls(
            base_url=env.pop("RHO_BASE_URL"),
            api_key=env.pop("RHO_API_KEY"),
            model=env.pop("RHO_MODEL"),
            system_prompt=system_prompt,
            effort=env.pop("RHO_EFFORT", ""),
            tools=[name for name in raw_tools.split(",") if name],
            subagents=env.pop("RHO_SUBAGENTS", "") == "1",
            context_budget_tokens=int(env.pop("RHO_CONTEXT_BUDGET_TOKENS", "150000")),
            max_turns=int(max_turns) if max_turns else None,
        )


def blocks_to_content(blocks: list) -> str | list[dict]:
    """One prompt turn's ACP content blocks as one chat message content. Text-only
    turns stay a plain string (transcript-friendly); images ride as data-URI parts."""
    parts: list[dict] = []
    for block in blocks:
        if isinstance(block, TextContentBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContentBlock):
            url = block.uri or f"data:{block.mime_type};base64,{block.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        else:
            kind = getattr(block, "type", type(block).__name__)
            raise TypeError(f"unsupported prompt content block: {kind!r}")
    if all(part["type"] == "text" for part in parts):
        return "\n\n".join(part["text"] for part in parts)
    return parts


class RhoAgent(Agent):
    """The driver's ACP face: one native session per process, one loop segment per
    prompt turn. The conversation accretes across turns — a later prompt lands as a
    user message on the same transcript, kernel, and named sessions."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.conn: Client | None = None
        self.driver: Driver | None = None
        self.messages: list[dict] = []

    def on_connect(self, conn: Client) -> None:
        self.conn = conn

    async def initialize(
        self, protocol_version, client_capabilities=None, client_info=None, **kwargs
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(image=True)
            ),
        )

    async def new_session(
        self, cwd, additional_directories=None, mcp_servers=None, **kwargs
    ) -> NewSessionResponse:
        mcp_config = {
            "mcpServers": {
                server.name: {"url": url}
                for server in mcp_servers or []
                if (url := getattr(server, "url", None))
            }
        }
        self.driver = Driver(self.config)
        await self.driver.setup(mcp_config)
        system = self.driver.system_prompt_base(
            self.driver.tools, advertise_agent=self.config.subagents
        )
        system += "\n\n" + TRANSCRIPT_SYSTEM_NOTE.format(path=TRANSCRIPT_PATH)
        if self.config.system_prompt:
            system += "\n\n" + self.config.system_prompt
        self.messages = [{"role": "system", "content": system}]
        self.driver.append_transcript(self.messages)
        self.driver.write_stats("ready")
        return NewSessionResponse(session_id="rho")

    async def prompt(self, session_id, prompt, **kwargs) -> PromptResponse:
        assert self.conn is not None and self.driver is not None
        self.messages.append({"role": "user", "content": blocks_to_content(prompt)})
        self.driver.append_transcript(self.messages)
        # The turn budget, not this call, is the authority on stopping.
        final = await self.driver.run_loop(
            self.messages, tools=self.driver.tools, depth=0, max_turns=None
        )
        self.driver.write_stats("segment")
        await self.conn.session_update(
            session_id=session_id,
            update=update_agent_message_text(
                final or "(the loop ended without a final reply)"
            ),
        )
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id, **kwargs) -> None:
        pass  # the verifiers runner never cancels a prompt turn


async def main() -> None:
    config = Config.from_env()
    write_diagnostics("started")
    agent = RhoAgent(config)
    try:
        # Serves the ACP connection over stdio until the client disconnects; stdout
        # is the protocol channel, so nothing else may print to it.
        await run_agent(agent)
    finally:
        if agent.driver is not None:
            await agent.driver.close()


if __name__ == "__main__":
    asyncio.run(main())
