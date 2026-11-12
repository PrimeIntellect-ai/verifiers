"""Typed native coding tools bound to the execution side of a split rollout.

Only the dispatcher runs on the host. File access and commands use the supplied
Runtime, so neither model arguments nor workspace files are evaluated on the host.
"""

import asyncio
import base64
import fnmatch
import json
import mimetypes
import shlex
import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from verifiers.v1.errors import HarnessError, SandboxError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.types import (
    ImageUrlContentPart,
    ImageUrlSource,
    ToolCall,
    ToolMessage,
)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, populate_by_name=True)


class _Shell(_Arguments):
    command: str
    cwd: str | None = Field(
        default=None, validation_alias=AliasChoices("cwd", "workdir")
    )
    timeout: float = Field(default=300, ge=0)
    timeoutMs: int | None = Field(default=None, gt=0)
    run_in_background: bool = Field(
        default=False, validation_alias=AliasChoices("run_in_background", "background")
    )


class _Path(_Arguments):
    path: str = Field(validation_alias=AliasChoices("path", "file_path", "filePath"))


class _Read(_Path):
    offset: int = Field(
        default=1, validation_alias=AliasChoices("offset", "line_offset")
    )
    limit: int = Field(
        default=2000, gt=0, validation_alias=AliasChoices("limit", "n_lines")
    )
    column_offset: int = Field(default=0, ge=0)
    max_chars: int = Field(default=100000, gt=0)


class _Write(_Path):
    content: str
    mode: Literal["overwrite", "append"] = "overwrite"


class _Edit(_Path):
    old: str = Field(
        min_length=1,
        validation_alias=AliasChoices("old_string", "oldText", "oldString", "old_str"),
    )
    new: str = Field(
        validation_alias=AliasChoices("new_string", "newText", "newString", "new_str")
    )
    replace_all: bool = Field(
        default=False, validation_alias=AliasChoices("replace_all", "replaceAll")
    )


class _Search(_Arguments):
    pattern: str
    path: str = "."
    glob: str | None = Field(
        default=None, validation_alias=AliasChoices("glob", "include")
    )
    caseSensitive: bool | None = None
    output_mode: Literal["content", "files_with_matches", "count", "count_matches"] = (
        "files_with_matches"
    )
    ignore_case: bool = Field(
        default=False, validation_alias=AliasChoices("-i", "ignoreCase", "ignore_case")
    )
    literal: bool = False
    multiline: bool = False
    context: int = Field(
        default=0, ge=0, validation_alias=AliasChoices("-C", "context")
    )
    before: int = Field(default=0, ge=0, validation_alias="-B")
    after: int = Field(default=0, ge=0, validation_alias="-A")
    limit: int = Field(
        default=250, ge=0, validation_alias=AliasChoices("head_limit", "limit")
    )
    offset: int = Field(default=0, ge=0)


class _List(_Arguments):
    path: str = "."
    limit: int = Field(default=500, gt=0)


class _Directory(_Arguments):
    cd: str | None = None


class _Task(_Arguments):
    task_id: str = Field(validation_alias=AliasChoices("task_id", "shell_id"))
    block: bool = False
    timeout: int = Field(default=30000, ge=0)


class _Todos(_Arguments):
    todos: list[dict]


class _Patch(_Arguments):
    patch: str = Field(validation_alias=AliasChoices("patchText", "patch", "input"))


class _Wait(_Arguments):
    timeout: int = Field(gt=0, le=90)
    task_id: str | None = None


@dataclass
class _Job:
    process: RuntimeProcess
    log: str
    done: asyncio.Task | None = None
    output: bytes = b""
    status: int | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_NATIVE_NAMES = {
    "claude_code": {
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "TaskOutput",
        "TaskStop",
        "TaskList",
        "KillShell",
        "TodoWrite",
        "TodoRead",
        "EnterPlanMode",
        "ExitPlanMode",
    },
    "kimi_code": {
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "ReadMediaFile",
        "TaskOutput",
        "TaskStop",
        "TaskList",
        "WaitFor",
    },
    "pi": {"bash", "read", "write", "edit", "grep", "find", "ls"},
    "deepseek": {"bash", "read_image", "working_directory"},
    "opencode": {
        "shell",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "patch",
        "todowrite",
    },
    "openclaw": {"read", "write", "edit", "apply_patch"},
}


# The worker receives validated JSON as an argv element, never shell source. It
# deliberately has no imports of verifiers, provider clients, or host credentials.
_FILES = r"""
import fnmatch, glob, json, os, re, sys
from pathlib import Path

operation, raw, cwd = sys.argv[1:]
a = json.loads(raw)
if cwd:
    os.chdir(cwd)
try:
    p = Path(a.get("path", "."))
    if operation == "read":
        lines = ([path.name + ("/" if path.is_dir() else "") + "\n" for path in sorted(p.iterdir())]
                 if p.is_dir() else p.read_text().splitlines(keepends=True))
        offset = a["offset"]
        if offset == 0:
            raise ValueError("line offsets are 1-based")
        start = max(0, offset - 1 if offset > 0 else len(lines) + offset)
        selected = lines[start:start + a["limit"]]
        if selected:
            selected[0] = selected[0][a["column_offset"]:]
        result = "".join(f"{i + start + 1}\t{line}" for i, line in enumerate(selected))[:a["max_chars"]]
    elif operation == "write":
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a" if a["mode"] == "append" else "w", newline="") as stream:
            stream.write(a["content"])
        result = f"Wrote {len(a['content'].encode())} bytes to {p}"
    elif operation == "edit":
        with p.open(newline="") as stream:
            original = stream.read()
        crlf = "\r\n" in original and "\n" not in original.replace("\r\n", "")
        content = original.replace("\r\n", "\n") if crlf else original
        count = content.count(a["old"])
        if count == 0 or (count != 1 and not a["replace_all"]):
            raise ValueError(f"old string must match {'at least once' if a['replace_all'] else 'exactly once'} (found {count})")
        updated = content.replace(a["old"], a["new"], -1 if a["replace_all"] else 1)
        with p.open("w", newline="") as stream:
            stream.write(updated.replace("\n", "\r\n") if crlf else updated)
        result = f"Edited {p}"
    elif operation == "glob":
        matches = sorted(str(p / path) for path in glob.glob(a["pattern"], root_dir=p, recursive=True) if (p / path).is_file())
        start, limit = a["offset"], a["limit"]
        result = "\n".join(matches[start:start + limit if limit else None])
    elif operation == "ls":
        result = "\n".join(sorted(path.name + ("/" if path.is_dir() else "") for path in p.iterdir())[:a["limit"]])
    elif operation == "grep":
        flags = (re.I if a["ignore_case"] else 0) | (re.S | re.M if a["multiline"] else 0)
        regex = re.compile(re.escape(a["pattern"]) if a["literal"] else a["pattern"], flags)
        result_lines = []
        for path in sorted([p] if p.is_file() else p.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            if a["glob"] and not fnmatch.fnmatch(str(path), a["glob"]):
                continue
            try:
                text = path.read_text()
            except (UnicodeError, OSError):
                continue
            matches = list(regex.finditer(text))
            if not matches:
                continue
            if a["output_mode"] == "files_with_matches":
                result_lines.append(str(path))
            elif a["output_mode"] in ("count", "count_matches"):
                result_lines.append(f"{path}:{len(matches)}")
            else:
                lines = text.splitlines()
                selected = set()
                for match in matches:
                    line = text.count("\n", 0, match.start())
                    end = text.count("\n", 0, match.end())
                    selected.update(range(max(0, line - max(a["before"], a["context"])), min(len(lines), end + max(a["after"], a["context"]) + 1)))
                result_lines.extend(f"{path}:{i + 1}:{lines[i]}" for i in sorted(selected))
        start, limit = a["offset"], a["limit"]
        result = "\n".join(result_lines[start:start + limit if limit else None])
    elif operation == "directory":
        if a["cd"] is not None:
            os.chdir(a["cd"])
        result = os.getcwd()
    elif operation == "patch":
        lines = a["patch"].strip().splitlines()
        if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
            raise ValueError("patch must start with *** Begin Patch and end with *** End Patch")
        pending = {}
        changes = []
        i = 1
        while i < len(lines) - 1:
            header = lines[i]
            i += 1
            if header.startswith("*** Add File: "):
                path = Path(header[len("*** Add File: "):])
                content = []
                while i < len(lines) - 1 and not lines[i].startswith("*** "):
                    if not lines[i].startswith("+"):
                        raise ValueError("added file lines must start with +")
                    content.append(lines[i][1:])
                    i += 1
                pending[path] = "\n".join(content) + "\n"
                changes.append(f"A {path}")
            elif header.startswith("*** Delete File: "):
                path = Path(header[len("*** Delete File: "):])
                if not path.is_file():
                    raise ValueError(f"file does not exist: {path}")
                pending[path] = None
                changes.append(f"D {path}")
            elif header.startswith("*** Update File: "):
                path = Path(header[len("*** Update File: "):])
                target = path
                if lines[i].startswith("*** Move to: "):
                    target = Path(lines[i][len("*** Move to: "):])
                    i += 1
                original = pending[path] if path in pending else path.read_text()
                if original is None:
                    raise ValueError(f"file was already deleted: {path}")
                content = original.splitlines()
                position = 0
                while i < len(lines) - 1 and not (lines[i].startswith("*** ") and lines[i] != "*** End of File"):
                    context = None
                    if lines[i].startswith("@@"):
                        context = lines[i][3:] if lines[i].startswith("@@ ") else None
                        i += 1
                    old, new = [], []
                    while i < len(lines) - 1 and not lines[i].startswith(("@@", "*** ")):
                        line = lines[i]
                        if not line or line[0] not in " +-":
                            raise ValueError("invalid patch update line")
                        if line[0] in " -":
                            old.append(line[1:])
                        if line[0] in " +":
                            new.append(line[1:])
                        i += 1
                    end = lines[i] == "*** End of File"
                    if end:
                        i += 1
                    if context is not None:
                        position = content.index(context, position) + 1
                    candidates = range(position, len(content) - len(old) + 1) if old else [len(content)]
                    found = next((start for start in candidates if content[start:start + len(old)] == old and (not end or start + len(old) == len(content))), None)
                    if found is None:
                        raise ValueError(f"patch context not found in {path}")
                    content[found:found + len(old)] = new
                    position = found + len(new)
                pending[target] = "\n".join(content) + "\n"
                if target != path:
                    pending[path] = None
                changes.append(f"M {target}")
            else:
                raise ValueError(f"invalid patch header: {header}")
        for path, content in pending.items():
            if content is None:
                path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
        result = "Success. Updated the following files:\n" + "\n".join(changes)
    else:
        raise ValueError("unknown file operation")
    if len(result) > 100000:
        result = result[:100000] + "\n[Output truncated; narrow the query or read a smaller range]"
    print(json.dumps({"content": result}))
except (OSError, ValueError, UnicodeError, re.error, IndexError, TypeError) as error:
    print(json.dumps({"content": f"error: {error}", "error": True}))
"""


@dataclass
class NativeRemoteTools:
    """Execution-side implementations of the core native coding tool vocabulary.

    Unknown tools fail closed instead of granting the supervisor permission to
    run them. Harness-specific model/orchestration tools require their own bridge.
    """

    runtime: Runtime
    harness: str
    disabled_tools: tuple[str, ...] = ()
    _cwd: str = ""
    _shell: RuntimeProcess | None = None
    _shell_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _shell_buffer: bytes = b""
    _jobs: dict[str, _Job] = field(default_factory=dict)
    _todos: list[dict] = field(default_factory=list)

    async def aclose(self) -> None:
        if self._shell is not None:
            await self._shell.terminate()
            self._shell = None
        for job in self._jobs.values():
            if job.status is None:
                await job.process.terminate()
            if job.done is not None:
                job.done.cancel()
        await asyncio.gather(
            *(job.done for job in self._jobs.values() if job.done is not None),
            return_exceptions=True,
        )

    async def _background(self, command: str, timeout: float) -> str:
        process = await self.runtime.exec(["bash", "-c", command], {})
        task_id = uuid.uuid4().hex
        job = _Job(process, f"/tmp/verifiers-background-{task_id}.log")
        self._jobs[task_id] = job
        await self.runtime.write(job.log, b"")

        async def collect(stream):
            async for data in stream:
                async with job.lock:
                    job.output = (job.output + data)[-100000:]
                    await self.runtime.write(job.log, job.output)

        async def complete():
            try:
                async with asyncio.timeout(timeout or None):
                    await asyncio.gather(
                        collect(process.stdout), collect(process.stderr)
                    )
                    job.status = await process.wait()
            except TimeoutError:
                await process.kill()
                job.status = await process.wait()
                job.output += b"\nerror: background command timed out"

        job.done = asyncio.create_task(complete())
        return f"task_id: {task_id}\nstatus: running\noutput_path: {job.log}\nRead output_path for the latest output (last 100000 bytes)"

    async def _persistent_bash(self, command: str, timeout: float | None) -> str:
        async with self._shell_lock:
            if self._shell is None or await self._shell.poll() is not None:
                self._shell = await self.runtime.exec(
                    [
                        "bash",
                        "--noprofile",
                        "--norc",
                        "-c",
                        "exec bash --noprofile --norc 2>&1",
                    ],
                    {},
                )
                if self._cwd:
                    await self._shell.write(
                        f"cd -- {shlex.quote(self._cwd)}\n".encode()
                    )
                self._shell_buffer = b""
            marker = "__vf_" + uuid.uuid4().hex + ":"
            await self._shell.write(
                f"eval -- {shlex.quote(command)}; printf '\\n{marker}%s\\n' \"$?\"\n".encode()
            )
            truncated = False
            try:
                async with asyncio.timeout(timeout):
                    while True:
                        marker_at = self._shell_buffer.find(marker.encode())
                        if marker_at >= 0:
                            end = self._shell_buffer.find(b"\n", marker_at)
                            if end >= 0:
                                content = (
                                    self._shell_buffer[:marker_at]
                                    .decode(errors="replace")
                                    .rstrip("\n")
                                )
                                status = self._shell_buffer[
                                    marker_at + len(marker) : end
                                ].decode()
                                self._shell_buffer = self._shell_buffer[end + 1 :]
                                if truncated:
                                    content = "[Earlier output truncated]\n" + content
                                return f"{content}\n[Command finished with exit code {status}]"
                        self._shell_buffer += await anext(self._shell.stdout)
                        if len(self._shell_buffer) > 1024 * 1024:
                            self._shell_buffer = self._shell_buffer[-1024 * 1024 :]
                            truncated = True
            except (TimeoutError, StopAsyncIteration):
                await self._shell.kill()
                self._shell = None
                return "error: persistent shell stopped or timed out; the next call starts a fresh shell"

    async def __call__(self, call: ToolCall) -> ToolMessage:
        permission = (
            "edit"
            if self.harness == "opencode" and call.name in {"write", "patch"}
            else call.name
        )
        if any(
            fnmatch.fnmatchcase(permission, pattern) for pattern in self.disabled_tools
        ):
            return ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=f"error: tool {call.name!r} is disabled",
            )
        name = call.name.lower()
        operations = {
            "bash": (_Shell, "bash"),
            "shell": (_Shell, "bash"),
            "read": (_Read, "read"),
            "read_file": (_Read, "read"),
            "write": (_Write, "write"),
            "write_file": (_Write, "write"),
            "edit": (_Edit, "edit"),
            "glob": (_Search, "glob"),
            "find": (_Search, "glob"),
            "grep": (_Search, "grep"),
            "ls": (_List, "ls"),
            "read_image": (_Path, "image"),
            "readmediafile": (_Path, "image"),
            "working_directory": (_Directory, "directory"),
            "taskoutput": (_Task, "task_output"),
            "taskstop": (_Task, "task_stop"),
            "killshell": (_Task, "task_stop"),
            "tasklist": (_Arguments, "task_list"),
            "todowrite": (_Todos, "todos"),
            "todo": (_Todos, "todos"),
            "todoread": (_Arguments, "todo_read"),
            "enterplanmode": (_Arguments, "plan"),
            "exitplanmode": (_Arguments, "plan"),
            "patch": (_Patch, "patch"),
            "apply_patch": (_Patch, "patch"),
            "waitfor": (_Wait, "wait"),
        }
        if name not in operations or call.name not in _NATIVE_NAMES.get(
            self.harness, set()
        ):
            return ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=f"error: split {self.harness} has no remote implementation for tool {call.name!r}",
            )
        model, operation = operations[name]
        try:
            args = model.model_validate_json(call.arguments or "{}")
        except ValidationError as error:
            content = (
                f"error: invalid {call.name} arguments ({error.errors()[0]['msg']})"
            )
        else:
            if (
                operation == "read"
                and isinstance(args, _Read)
                and (mimetypes.guess_type(args.path)[0] or "").startswith("image/")
            ):
                operation = "image"
            if operation == "bash":
                assert isinstance(args, _Shell)
                # Claude's timeout is milliseconds; Pi and Kimi use seconds.
                timeout = (
                    args.timeoutMs / 1000
                    if args.timeoutMs is not None
                    else args.timeout
                    / (
                        1000
                        if self.harness in {"claude_code", "opencode"}
                        and "timeout" in args.model_fields_set
                        else 1
                    )
                )
                command = args.command
                cwd = args.cwd or self._cwd
                if cwd:
                    command = f"cd -- {shlex.quote(cwd)} && {command}"
                if args.run_in_background:
                    content = await self._background(command, timeout)
                elif self.harness == "deepseek":
                    content = await self._persistent_bash(args.command, timeout or None)
                else:
                    try:
                        async with asyncio.timeout(timeout or None):
                            result = await self.runtime.run(["bash", "-c", command], {})
                        content = result.stdout + result.stderr
                        if result.exit_code:
                            content += f"\n[Command finished with exit code {result.exit_code}]"
                    except TimeoutError:
                        content = f"error: command timed out after {timeout:g} seconds"
            elif operation in {"task_output", "task_stop"}:
                assert isinstance(args, _Task)
                job = self._jobs.get(args.task_id)
                if job is None:
                    content = f"error: task {args.task_id!r} not found"
                else:
                    if operation == "task_stop" and job.status is None:
                        await job.process.kill()
                    if args.block or operation == "task_stop":
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(job.done), args.timeout / 1000
                            )
                        except TimeoutError:
                            pass
                    content = f"task_id: {args.task_id}\nstatus: {'running' if job.status is None else 'completed'}\nexit_code: {job.status}\n{job.output.decode(errors='replace')}"
            elif operation == "task_list":
                content = json.dumps(
                    {
                        key: {
                            "status": "running" if job.status is None else "completed",
                            "exit_code": job.status,
                        }
                        for key, job in self._jobs.items()
                    }
                )
            elif operation == "wait":
                assert isinstance(args, _Wait)
                jobs = (
                    ([self._jobs[args.task_id]] if args.task_id in self._jobs else [])
                    if args.task_id
                    else list(self._jobs.values())
                )
                pending = [
                    job.done
                    for job in jobs
                    if job.done is not None and not job.done.done()
                ]
                if pending:
                    await asyncio.wait(
                        pending,
                        timeout=args.timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                content = json.dumps(
                    {
                        key: {
                            "status": "running" if job.status is None else "completed",
                            "exit_code": job.status,
                        }
                        for key, job in self._jobs.items()
                        if args.task_id is None or key == args.task_id
                    }
                )
            elif operation == "todos":
                assert isinstance(args, _Todos)
                self._todos = args.todos
                content = json.dumps(self._todos)
            elif operation == "todo_read":
                content = json.dumps(self._todos)
            elif operation == "plan":
                content = (
                    "Plan mode enabled"
                    if name == "enterplanmode"
                    else "Plan mode finished"
                )
            elif operation == "image":
                assert isinstance(args, _Path)
                path = args.path
                if self._cwd and not path.startswith("/"):
                    path = f"{self._cwd}/{path}"
                mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
                if not mime.startswith("image/"):
                    content = "error: the requested file is not a supported image"
                else:
                    try:
                        data = await self.runtime.read(path, max_bytes=20 * 1024 * 1024)
                    except (OSError, SandboxError) as error:
                        content = f"error: {error}"
                    else:
                        content = [
                            ImageUrlContentPart(
                                image_url=ImageUrlSource(
                                    url=f"data:{mime};base64,{base64.b64encode(data).decode()}"
                                )
                            )
                        ]
            else:
                if isinstance(args, _Search):
                    if name == "find" and "/" not in args.pattern:
                        args.pattern = "**/" + args.pattern
                    if args.caseSensitive is not None:
                        args.ignore_case = not args.caseSensitive
                    if (
                        self.harness in {"pi", "opencode"}
                        and "output_mode" not in args.model_fields_set
                    ):
                        args.output_mode = "content"
                result = await self.runtime.run(
                    [
                        "python3",
                        "-c",
                        _FILES,
                        operation,
                        args.model_dump_json(),
                        self._cwd,
                    ],
                    {},
                )
                if result.exit_code:
                    raise HarnessError(
                        f"split {self.harness} file worker failed: {result.stderr}"
                    )
                response = json.loads(result.stdout)
                content = response["content"]
                if operation == "directory" and not response.get("error"):
                    self._cwd = content
        return ToolMessage(tool_call_id=call.id, name=call.name, content=content)
