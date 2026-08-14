# /// script
# requires-python = ">=3.10"
# dependencies = [
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

The transcript is append-only between compaction boundaries: assistant messages (reasoning
included) are re-sent complete, and the only rewriting event is Codex-style checkpoint
compaction — triggered by the token budget, a context overflow, or the model's own
`compact` tool call.

Deliberate absences (known, may be wanted later, not built yet):
- Parallel tool calls: multiple calls in one turn execute sequentially, and the kernel
  bridge is a serial stdio channel — code-side `gather` over tools needs bridge
  multiplexing (frame ids), one coherent upgrade with interception-side concurrency.
- Background tools: no `background`/yield machinery (Codex's session+wait stack); the
  documented answer for long-running processes is bash: nohup, redirect to a file, read.
"""

import argparse
import asyncio
import base64
import contextlib
import itertools
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
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
    b"RIFF": "image/webp",
}
MAX_IMAGE_BYTES = 5 * 1024 * 1024

SPAWN_RESERVE_TURNS = 3
"""Refuse to delegate when fewer turns than this remain: a subagent can spend the whole
shared budget, but the parent must keep enough to act on the result. A reserve, not a
cap — the box imposes no limits of its own; the framework's turn budget is the bound."""

TUNNEL_TIMEOUT = httpx.Timeout(600.0, connect=30.0)
TUNNEL_POOL = httpx.Limits(max_connections=16, max_keepalive_connections=16, keepalive_expiry=900.0)
"""Everything the box talks to is the host reached back through the runtime's tunnel,
where opening a connection is the slow part (measured 3-16s, up to 55s). Pool for the
whole rollout so that cost is paid per rollout, not per call."""

MCP_CALL_ATTEMPTS = 6
TOOLS_DIR = "tools"
DISCOVERY_CATALOG = "tools_catalog"
RUNTIME_DIAGNOSTICS = "/tmp/.rho-runtime.json"

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

# With a history file, the summary is licensed to be tight: raw data stays greppable, so
# the checkpoint carries decisions and state instead of hedging with carried payloads.
HISTORY_PROMPT_NOTE = (
    "\nThe full transcript you are summarizing will remain available to the next LLM in a "
    "file it can grep, so keep the summary tight and targeted — decisions, state, and next "
    "steps. Raw data, long outputs, and exact quotes can be recovered from the file."
)
HISTORY_FRAMING_NOTE = (
    "\n\nThe full transcript this summary replaced is at {path} — grep it (bash) if a "
    "detail you need is missing from the summary. Work from evidence: the workspace is "
    "authoritative — inspect the current state of files before relying on the summary "
    "or the history."
)

RESUME_NOTE = (
    "This session continues an earlier exchange: run_code variables from previous "
    "segments are gone; files, ./tools/, and /tmp/.rho artifacts are intact."
)


def assemble_messages(system: str, prompt: str, initial: list[dict] | None) -> list[dict]:
    """The loop's starting transcript. A resumed segment replays the exchange's
    conversation: a LEADING system message is the previous segment's harness-built
    system prompt and is replaced by the fresh rebuild; any later explicit system
    messages survive (the base `Harness.resume` contract). A fresh segment opens
    with the task prompt."""
    head = {"role": "system", "content": system}
    if initial is not None:
        if initial and initial[0].get("role") == "system":
            initial = initial[1:]
        return [head, *initial]
    return [head, {"role": "user", "content": prompt}]


def render_history(messages: list[dict]) -> str:
    """Render discarded transcript messages as greppable role-tagged text."""
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
    payload = {"phase": phase, "python": platform.python_version(), **values}
    Path(RUNTIME_DIAGNOSTICS).write_text(json.dumps(payload, sort_keys=True))


# --------------------------------------------------------------------------------------
# Output bounding: three ceilings, tail-kept, spill to a named path
# --------------------------------------------------------------------------------------

_spill_counter = 0


def _spill(text: str, label: str) -> str:
    # mkstemp, not a counter: the counter resets per process, and a resumed segment
    # (or a shared /tmp) would overwrite files an earlier transcript still names.
    SPILL_DIR.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=SPILL_DIR, prefix=f"{label}-", suffix=".txt")
    os.write(fd, text.encode())
    os.close(fd)
    return path


def cap_output(text: str, label: str = "output") -> str:
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
        return f"{path} is a directory ({len(entries)} entries) — use bash ls; first entries: " + ", ".join(entries[:20])
    with p.open("rb") as f:
        head = f.read(8)
    for magic, mime in IMAGE_MAGIC.items():
        if head.startswith(magic):
            size = p.stat().st_size
            if size > MAX_IMAGE_BYTES:
                return f"{path} is a {mime} of {size} bytes — over the {MAX_IMAGE_BYTES} byte image cap"
            READ_LEDGER.add(str(p))
            encoded = base64.b64encode(p.read_bytes()).decode()
            return [
                {"type": "text", "text": f"{path} ({mime}, {size} bytes)"},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
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
                line = line[:MAX_LINE_CHARS] + " …[line clipped, use bash to see the rest]"
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
        return body + f"\n[Showing lines {offset}-{end}; more of the file follows. Use offset={resume} to continue.]"
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
        original = p.read_text()
    except Exception as e:  # noqa: BLE001 - tool failures are returned to the model
        return f"could not read {path}: {e}"

    spans: list[tuple[int, int, str]] = []
    for i, edit in enumerate(edits):
        old, new = edit.get("oldText"), edit.get("newText")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return f"edits[{i}]: oldText must be a non-empty string and newText a string"
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
        p.write_text("".join(result))
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
        out, _ = proc.communicate()
        return cap_output((out or "") + f"\n[command timed out after {timeout:g}s and was killed]", "bash")
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

KERNEL_ENV_KEEP = ("PATH", "HOME", "LANG", "TERM", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV")
# No UV_ prefix: UV_INDEX_<NAME>_PASSWORD and credential-bearing index URLs ride it.
KERNEL_ENV_PREFIXES = ("LC_", "XDG_")


def kernel_env() -> dict[str, str]:
    """Allowlisted env for the kernel: model code gets capabilities, not credentials."""
    return {
        k: v
        for k, v in os.environ.items()
        if k in KERNEL_ENV_KEEP or k.startswith(KERNEL_ENV_PREFIXES)
    }


class Kernel:
    """Driver side of the kernel: lifecycle, cells, and the paused-clock timeout."""

    def __init__(self, init_payload: dict):
        self._init_payload = init_payload
        self._proc: asyncio.subprocess.Process | None = None
        self._runner_path: str | None = None
        self._death_note = ""

    async def _ensure(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        if self._runner_path is None:
            fd, self._runner_path = tempfile.mkstemp(prefix=".rho-runner-", suffix=".py")
            os.write(fd, RUNNER_SOURCE.encode())
            os.close(fd)
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-u",
            self._runner_path,
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
                return restart_note + "[kernel died running this cell — fresh kernel next call: variables lost, files intact.]"
            if "fatal" in frame:
                self._mark_dead("kernel crashed")
                return restart_note + cap_output(f"[kernel crashed]\n{frame['fatal']}", "run_code")
            if "call" in frame:
                # Bridge time is the tool's, not the cell's: clock paused.
                try:
                    result = await handle_call(frame["call"])
                    self._send({"result": result})
                except BridgeError as e:
                    self._send({"error": {"type": e.type, "message": str(e)}})
                except Exception as e:  # noqa: BLE001 - bridge failures raise inside the kernel
                    self._send({"error": {"type": "RuntimeError", "message": f"{type(e).__name__}: {e}"}})
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
                frame = await self._read_frame(timeout=max(0.1, deadline - time.monotonic()))
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
        read, write, *_ = await stack.enter_async_context(streamable_http_client(url, http_client=http_client))
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
        *(with_retry(lambda name=name, spec=spec: list_tools(name, spec)) for name, spec in servers)
    )
    for (name, _spec), tools in zip(servers, listed):
        for tool in tools:
            if tool.name == DISCOVERY_CATALOG:
                # The box's own channel: read once at startup, never bound or documented.
                catalog_server = (name, tool.name)
                continue
            # Wire contract with task authors: tools are named <app>_<verb> (first
            # underscore splits); a verb-less name lands in the shared `misc` app.
            bare = tool.name.split("_", 1)[1] if tool.name.startswith("tools_") else tool.name
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
        async with mcp_session(clients[server_name], servers[server_name]["url"]) as session:
            return await session.call_tool(raw_name, arguments)

    result = await with_retry(call)
    text = mcp_content(result)
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001 - non-JSON results pass through as text
        return text


def _safe_component(name: str) -> str:
    """Doc-tree names come from the task's MCP server; keep them single path components."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"unsafe tool doc path component: {name!r}")
    return name


def write_tool_docs(box: Path, docs: dict) -> None:
    for (app, verb), text in docs.items():
        d = box / TOOLS_DIR / _safe_component(app)
        d.mkdir(parents=True, exist_ok=True)
        (d / _safe_component(verb)).write_text(text)
    if docs:
        app, verb = next(iter(docs))
        (box / TOOLS_DIR / "README").write_text(
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
    root = box / TOOLS_DIR
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
# Tool schemas (wire surface): five small tools, conditionally advertised
# --------------------------------------------------------------------------------------


def build_tool_schemas(tools: list[str]) -> list[dict]:
    def fn(name, description, properties, required):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
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
            "Write a file (creates parent directories, overwrites).",
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
                        "properties": {"oldText": {"type": "string"}, "newText": {"type": "string"}},
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
                "timeout": {"type": "number", "description": "seconds (optional, no default)"},
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
    return f"\n\n{lead} only valid JSON matching this schema (no prose, no code fences):\n" + json.dumps(schema)


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

    def __init__(self, args):
        self.args = args
        self.client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            http_client=httpx.AsyncClient(timeout=TUNNEL_TIMEOUT, limits=TUNNEL_POOL),
        )
        self.budget = TurnBudget(args.max_turns)
        self.box = Path.cwd()
        self.tools: list[str] = args.tools.split(",") if args.tools else []
        self.kernel: Kernel | None = None
        self.apps_payload: dict = {}
        self.dispatch: dict = {}
        self.servers: dict = {}
        self.clients: dict = {}
        self.documented_apps: set[str] = set()
        self.tool_discovery: dict[str, int] = {}
        self.subagents_spawned = 0
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

    async def setup(self) -> None:
        config = json.loads(self.args.mcp_config) if self.args.mcp_config else {}
        self.servers = config.get("mcpServers", {})
        self.clients = mcp_clients(config)
        self.dispatch, docs, catalog_ref = await connect_mcp(config, self.clients)
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
            raise ValueError(f"app names collide with run_code globals: {sorted(reserved)}")
        self.apps_payload = apps_payload
        if "run_code" in self.tools:
            self.kernel = Kernel(
                {"apps": apps_payload, "subagents": self.args.subagents}
            )

    # --- bridge handlers ----------------------------------------------------------

    async def handle_call(self, call: dict, depth: int):
        kind = call.get("kind")
        if kind == "mcp":
            return await self.do_mcp(call.get("app", ""), call.get("verb", ""), call.get("args") or {})
        if kind == "completion":
            return await self.do_completion(call.get("prompt", ""), call.get("schema"), call.get("system"))
        if kind == "agent":
            if not self.args.subagents:
                raise BridgeError("NameError", "agent() is not enabled in this session")
            if depth > 0:
                raise BridgeError("RuntimeError", "subagents cannot spawn subagents (depth 1)")
            return await self.do_agent(
                call.get("prompt", ""), call.get("schema"), call.get("effort"),
                call.get("tools"), call.get("session"),
            )
        raise BridgeError("RuntimeError", f"unknown bridge call {kind!r}")

    async def do_mcp(self, app: str, verb: str, arguments: dict):
        entry = self.dispatch.get((app, verb))
        if entry is None:
            if app in self.served_apps:
                raise BridgeError(
                    "AttributeError", f"{app} has no {verb!r} tool — its tools are documented in ./tools/{app}/"
                )
            if app in self.documented_apps:
                self.tool_discovery["decoy_calls"] = self.tool_discovery.get("decoy_calls", 0) + 1
                raise BridgeError(
                    "NameError",
                    f"no such app in this workspace: {app!r} — it is documented, but this workspace "
                    "has no such integration; the work has to go through an app that is connected",
                )
            raise BridgeError(
                "NameError", f"no tool named {app}.{verb!r} — explore ./tools/<app>/ and call <app>.<verb>(...)"
            )
        server, raw, _ = entry
        return await call_mcp_raw(self.servers, self.clients, server, raw, arguments)

    async def do_completion(self, prompt: str, schema, system):
        if not self.budget.take():
            raise BridgeError("RuntimeError", "turn budget exhausted — no model calls left")
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        body = str(prompt) + (schema_suffix("Respond with", schema) if schema is not None else "")
        messages.append({"role": "user", "content": body})
        problem = "no attempts left"
        for attempt in range(2):
            if attempt and not self.budget.take():
                break
            completion = await self.client.chat.completions.create(
                model=self.args.model, messages=messages, **self._effort_kwargs()
            )
            text = completion.choices[0].message.content or ""
            if schema is None:
                return text
            value, problem = parse_reply(text, schema)
            if problem is None:
                return value
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"{problem}. Reply with only the corrected JSON."})
        raise BridgeError("ValueError", f"completion did not return JSON matching the schema: {problem}")

    async def do_agent(self, prompt: str, schema, effort, tools, session=None):
        remaining = self.budget.remaining
        if remaining is not None and remaining < SPAWN_RESERVE_TURNS:
            raise BridgeError(
                "RuntimeError",
                f"not enough turn budget left to delegate ({remaining} turns remain) — "
                "finish the task directly",
            )
        self.subagents_spawned += 1
        body = str(prompt) + (schema_suffix("Your final message must be", schema) if schema is not None else "")
        if session and session in self.sessions:
            # Continue the named session: same transcript, tool surface, and kernel —
            # the follow-up rides the context already paid for.
            stored = self.sessions[session]
            messages, allowed, kernel = stored["messages"], stored["tools"], stored["kernel"]
            messages.append({"role": "user", "content": body})
        else:
            allowed = [t for t in self.tools if t != "compact"]
            if tools is not None:
                unknown = [t for t in tools if t not in allowed]
                if unknown:
                    raise BridgeError("ValueError", f"unknown subagent tools: {unknown} (available: {allowed})")
                allowed = [t for t in allowed if t in tools]
            # A subagent gets its own kernel: agent() is only reachable from inside a
            # parent cell, whose kernel is blocked in bridge() — routing sub-cells to
            # it would corrupt the frame stream. Blank context means blank namespace.
            kernel = Kernel({"apps": self.apps_payload, "subagents": False}) if "run_code" in allowed else None
            messages = [
                {"role": "system", "content": self.subagent_system_prompt(allowed)},
                {"role": "user", "content": body},
            ]
            if session:
                self.sessions[session] = {"messages": messages, "tools": allowed, "kernel": kernel}
        try:
            final = await self.run_loop(
                messages,
                tools=allowed,
                depth=1,
                # No per-spawn ceiling: allocation is the policy's call; the shared budget
                # binds, minus the reserve that keeps the parent able to act.
                max_turns=None if self.budget.remaining is None else self.budget.remaining - (SPAWN_RESERVE_TURNS - 1),
                effort=effort,
                kernel=kernel,
            )
            if final is None:
                return None
            if schema is None:
                return final
            problem = None
            for _ in range(2):
                value, problem = parse_reply(final, schema)
                if problem is None:
                    return value
                if self.budget.remaining is not None and self.budget.remaining < 1:
                    break
                messages.append({"role": "user", "content": f"{problem}. Reply with only the corrected JSON."})
                final = await self.run_loop(messages, tools=[], depth=1, max_turns=1, effort=effort, kernel=kernel)
                if final is None:
                    break
            if final is not None:
                value, problem = parse_reply(final, schema)
                if problem is None:
                    return value
            raise BridgeError("ValueError", f"subagent did not return JSON matching the schema: {problem}")
        finally:
            if kernel is not None and not session:
                await kernel.close()

    # --- native tool dispatch -------------------------------------------------------

    async def execute_tool(self, name: str, arguments: dict, depth: int, kernel: Kernel | None = None):
        if name == "read":
            return await asyncio.to_thread(
                run_read, arguments.get("path", ""), arguments.get("offset"), arguments.get("limit"), self.box
            )
        if name == "write":
            return await asyncio.to_thread(
                run_write, arguments.get("path", ""), arguments.get("content", ""), self.box
            )
        if name == "edit":
            return await asyncio.to_thread(
                run_edit, arguments.get("path", ""), arguments.get("edits"), self.box
            )
        if name == "bash":
            return await asyncio.to_thread(
                run_bash, arguments.get("command", ""), arguments.get("timeout"), self.box
            )
        if name == "run_code":
            kernel = kernel if kernel is not None else self.kernel
            assert kernel is not None
            return await kernel.run_cell(
                arguments.get("code", ""), lambda call: self.handle_call(call, depth=depth)
            )
        if name == "compact" and depth == 0:
            self.compaction_pending = True
            return "[context checkpoint scheduled before your next turn]"
        return f"unknown tool {name}"

    # --- prompts --------------------------------------------------------------------

    def system_prompt_base(self, tools: list[str], advertise_agent: bool | None = None) -> str:
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
            lines.append("`completion(prompt, schema=None)` makes one standalone model call from code.")
            if self.args.subagents if advertise_agent is None else advertise_agent:
                lines.append(
                    "`agent(prompt, schema=None, effort=None, tools=None, session=None)` runs a "
                    "subagent with a blank context and returns its final message (parsed when "
                    "schema is given); give it complete, self-contained instructions. Pass a "
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

    async def checkpoint(self, messages: list[dict], tool_schemas: list[dict]) -> None:
        keep = 2  # the protected prefix: system + the opening user message (assemble_messages)
        history_path = _spill(render_history(messages[keep:]), "history")
        request = [*messages, {"role": "user", "content": COMPACTION_PROMPT + HISTORY_PROMPT_NOTE}]
        while True:
            try:
                completion = await self.client.chat.completions.create(
                    model=self.args.model,
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
                    while keep < len(request) - 1 and request[keep].get("role") == "tool":
                        del request[keep]
                    continue
                raise
        summary = completion.choices[0].message.content or "(no summary available)"
        usage = getattr(completion, "usage", None)
        if usage and usage.prompt_tokens:
            self.peak_prompt_tokens = max(self.peak_prompt_tokens, usage.prompt_tokens)
        framing = COMPACTION_FRAMING + "\n\n" + summary + HISTORY_FRAMING_NOTE.format(path=history_path)
        messages[:] = [*messages[:keep], {"role": "user", "content": framing}]
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
            and self.args.context_budget_tokens
            and self.last_prompt_tokens >= self.args.context_budget_tokens
        )

    # --- the loop ----------------------------------------------------------------------

    def _effort_kwargs(self, effort: str | None = None) -> dict:
        value = effort if effort is not None else self.args.effort
        return {"reasoning_effort": value} if value else {}

    def _notices(self) -> str:
        """Root-loop notices, appended to the turn's last tool result — the observation
        channel, never the system prompt, so every request extends the one before."""
        notices = []
        if self.args.disclose_budget and self.budget.max_turns is not None:
            spent = self.budget.spent
            notices.append(
                f"[harness] Turn budget: {spent}/{self.budget.max_turns} used, "
                f"{self.budget.max_turns - spent} remaining."
            )
        if "compact" in self.tools and self.last_prompt_tokens:
            notices.append(
                f"[harness] Context: ~{self.last_prompt_tokens} prompt tokens "
                f"(checkpoint threshold {self.args.context_budget_tokens})."
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
                if not self.budget.take():
                    break
                await self.checkpoint(messages, tool_schemas)
            if not self.budget.take():
                break
            try:
                completion = await self.client.chat.completions.create(
                    model=self.args.model,
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
            if usage and usage.prompt_tokens and depth == 0:
                self.last_prompt_tokens = usage.prompt_tokens
                self.peak_prompt_tokens = max(self.peak_prompt_tokens, usage.prompt_tokens)
                if self.just_compacted:
                    # The first work turn after a checkpoint is the effectiveness probe:
                    # still over the threshold means the protected prefix itself is too
                    # big — further automatic compaction would loop without shrinking.
                    self.just_compacted = False
                    if (
                        self.args.context_budget_tokens
                        and usage.prompt_tokens >= self.args.context_budget_tokens
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
                        out = await self.execute_tool(tc.function.name, arguments, depth, kernel)
                    except BridgeError as e:
                        out = f"{tc.function.name} failed: {e}"
                    except Exception as e:  # noqa: BLE001 - tool faults are observations, not rollout failures
                        out = f"{tc.function.name} failed: {type(e).__name__}: {e}"
                content = out if isinstance(out, list) else str(out)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            if depth == 0:
                if isinstance(messages[-1]["content"], str):
                    messages[-1]["content"] += self._notices()
                self.write_stats("loop")
        return final

    def write_stats(self, phase: str) -> None:
        write_diagnostics(
            phase,
            turns=self.budget.spent,
            compactions=self.compactions,
            peak_prompt_tokens=self.peak_prompt_tokens,
            subagents=self.subagents_spawned,
            **({"tool_discovery": dict(self.tool_discovery)} if self.tool_discovery else {}),
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--system-prompt", default="")
    p.add_argument("--prompt", default="")
    p.add_argument("--initial-messages-file", default="")
    p.add_argument("--mcp-config", default="")
    p.add_argument("--effort", default="")
    p.add_argument("--tools", default="read,write,edit,bash,run_code")
    p.add_argument("--subagents", action="store_true")
    p.add_argument("--context-budget-tokens", type=int, default=150_000)
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--disclose-budget", action="store_true")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    write_diagnostics("started")
    driver = Driver(args)
    try:
        await driver.setup()
        initial = None
        if args.initial_messages_file:
            path = Path(args.initial_messages_file)
            initial = json.loads(path.read_text())
            path.unlink(missing_ok=True)
        system = driver.system_prompt_base(driver.tools)
        if initial is not None and "run_code" in driver.tools:
            system += "\n\n" + RESUME_NOTE
        if args.system_prompt:
            system += "\n\n" + args.system_prompt
        messages = assemble_messages(system, args.prompt, initial)
        # The turn budget, not this call, is the authority on stopping.
        await driver.run_loop(messages, tools=driver.tools, depth=0, max_turns=None)
        driver.write_stats("complete")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
