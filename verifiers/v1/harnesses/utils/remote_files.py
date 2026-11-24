"""Native coding tool operations on the bound workspace's files."""

import asyncio
import fnmatch
from functools import cache
from pathlib import PurePosixPath

from verifiers.v1.harnesses.utils import remote_search
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import collect_process

MAX_FILE_BYTES = 16 * 1024 * 1024
TRUNCATED = "\n[Output truncated; narrow the query or read a smaller range]"


async def run_shell(
    runtime: Runtime, command: str, env: dict[str, str], *, shell: str = "bash"
):
    """Run a shell command, piping oversized scripts on stdin to avoid MAX_ARG_STRLEN."""
    if len(command.encode()) <= 65536:
        return await runtime.run([shell, "-c", command], env)
    process = await runtime.exec([shell, "-s"], env)
    drain = asyncio.create_task(collect_process(process))
    try:
        data = command.encode()
        for offset in range(0, len(data), 65536):
            await process.write(data[offset : offset + 65536])
        if not data.endswith(b"\n"):
            await process.write(b"\n")
        await process.write(b"\nexit $?\n")
        return await drain
    finally:
        if not drain.done():
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)


async def _command(runtime, *argv):
    result = await runtime.run(list(argv), {})
    if result.exit_code:
        raise ValueError(
            result.stderr.strip() or f"{argv[0]} exited with status {result.exit_code}"
        )
    return result.stdout


async def read_text(runtime, path, *, normalize=False):
    data = await runtime.read(str(path), max_bytes=MAX_FILE_BYTES)
    text = data.decode()
    return text.replace("\r\n", "\n").replace("\r", "\n") if normalize else text


async def _append(runtime, path, data):
    script = 'mkdir -p -- "$(dirname -- "$1")" && head -c "$2" >> "$1"'
    process = await runtime.exec(
        ["sh", "-c", script, "append", str(path), str(len(data))], {}
    )
    drain = asyncio.create_task(collect_process(process))
    try:
        for offset in range(0, len(data), 65536):
            await process.write(data[offset : offset + 65536])
        result = await drain
        if result.exit_code:
            raise ValueError(result.stderr.strip())
    finally:
        if not drain.done():
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)


async def _names(runtime, script, *args):
    # Keep arbitrary filenames (including embedded newlines) separate and cap listing transfers.
    cmd = "set -o pipefail; { " + script + "; } | head -c 8388609"
    result = await runtime.run(["bash", "-c", cmd, "list", *args], {})
    if len(result.stdout.encode()) > 8 * 1024 * 1024:
        raise ValueError("listing exceeds 8 MiB; narrow the path or glob")
    if result.exit_code:
        raise ValueError(result.stderr.strip() or "could not list workspace files")
    return result.stdout.removesuffix("\0").split("\0") if result.stdout else []


_LIST = r"""cd -- "$1" || exit 1; shopt -s nullglob dotglob
for f in *; do if [[ -d $f ]]; then printf '%s/\0' "$f"; else printf '%s\0' "$f"; fi; done"""


async def _glob(runtime, cwd, pattern):
    parts = PurePosixPath(pattern).parts
    first = next(
        (i for i, part in enumerate(parts) if any(c in part for c in "*?[")), len(parts)
    )
    prefix = PurePosixPath(*parts[:first])
    if first == len(parts):
        exists = await runtime.run(["test", "-f", str(PurePosixPath(cwd) / prefix)], {})
        return [str(prefix)] if exists.exit_code == 0 else []
    patterns = parts[first:]

    @cache
    def matches(path, index):
        if index == len(patterns):
            return not path
        pattern = patterns[index]
        if pattern == "**":
            return matches(path, index + 1) or bool(
                path and not path[0].startswith(".") and matches(path[1:], index)
            )
        return bool(
            path
            and (not path[0].startswith(".") or pattern.startswith("."))
            and fnmatch.fnmatchcase(path[0], pattern)
            and matches(path[1:], index + 1)
        )

    script = 'cd -- "$1" || exit 0; [[ -d $2 ]] || exit 0; find -L "$2" "${@:3}" -type f -print0'
    depth = [] if "**" in patterns else ["-maxdepth", str(len(patterns))]
    root = str(prefix) if prefix.is_absolute() else "./" + str(prefix)
    names = await _names(runtime, script, cwd, root, *depth)
    result = []
    for name in names:
        if matches(PurePosixPath(name).relative_to(prefix).parts, 0):
            result.append(name)
        matches.cache_clear()
    return result


async def _patch(runtime, cwd, patch):
    lines = patch.strip().splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError(
            "patch must start with *** Begin Patch and end with *** End Patch"
        )
    pending = {}
    changes = []
    i = 1
    while i < len(lines) - 1:
        header = lines[i]
        i += 1
        if header.startswith("*** Add File: "):
            path = PurePosixPath(header[len("*** Add File: ") :])
            content = []
            while i < len(lines) - 1 and not lines[i].startswith("*** "):
                if not lines[i].startswith("+"):
                    raise ValueError("added file lines must start with +")
                content.append(lines[i][1:])
                i += 1
            pending[path] = "\n".join(content) + "\n"
            changes.append(f"A {path}")
        elif header.startswith("*** Delete File: "):
            path = PurePosixPath(header[len("*** Delete File: ") :])
            if (await runtime.run(["test", "-f", str(cwd / path)], {})).exit_code:
                raise ValueError(f"file does not exist: {path}")
            pending[path] = None
            changes.append(f"D {path}")
        elif header.startswith("*** Update File: "):
            path = PurePosixPath(header[len("*** Update File: ") :])
            target = path
            if lines[i].startswith("*** Move to: "):
                target = PurePosixPath(lines[i][len("*** Move to: ") :])
                i += 1
            original = (
                pending[path]
                if path in pending
                else await read_text(runtime, cwd / path, normalize=True)
            )
            if original is None:
                raise ValueError(f"file was already deleted: {path}")
            content = original.splitlines()
            position = 0
            while i < len(lines) - 1 and not (
                lines[i].startswith("*** ") and lines[i] != "*** End of File"
            ):
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
                candidates = (
                    range(position, len(content) - len(old) + 1)
                    if old
                    else [len(content)]
                )
                found = next(
                    (
                        start
                        for start in candidates
                        if content[start : start + len(old)] == old
                        and (not end or start + len(old) == len(content))
                    ),
                    None,
                )
                if found is None:
                    raise ValueError(f"patch context not found in {path}")
                content[found : found + len(old)] = new
                position = found + len(new)
            pending[target] = "\n".join(content) + "\n"
            if target != path:
                pending[path] = None
            changes.append(f"M {target}")
        else:
            raise ValueError(f"invalid patch header: {header}")
    for path, updated in pending.items():
        if updated is None:
            await _command(runtime, "rm", "--", str(cwd / path))
        else:
            await runtime.write(str(cwd / path), updated.encode())
    return "Success. Updated the following files:\n" + "\n".join(changes)


async def _read_lines(runtime, target, args):
    offset = args["offset"]
    if not offset:
        raise ValueError("line offsets are 1-based")
    kind = await _command(
        runtime,
        "sh",
        "-c",
        'if [ -d "$1" ]; then echo dir; else wc -c < "$1"; fi',
        "stat",
        target,
    )
    if kind.strip() == "dir":
        names = await _names(runtime, _LIST, target)
        lines = [
            name + "\n"
            for name in sorted(names, key=lambda name: name.removesuffix("/"))
        ]
    elif int(kind) <= MAX_FILE_BYTES:
        lines = (await read_text(runtime, target, normalize=True)).splitlines(
            keepends=True
        )
    else:
        # A small window into a large log should transfer only that window from B.
        if offset < 0:
            count = await _command(
                runtime, "sh", "-c", """awk 'END {print NR}' < "$1" """, "lines", target
            )
            offset = max(1, int(count) + offset + 1)
        end = offset + args["limit"] - 1
        # Four bytes cover one UTF-8 code point. Read enough to apply the first-line column offset.
        budget = min(
            MAX_FILE_BYTES,
            4 * (min(args["max_chars"], 100001) + args["column_offset"] + 1),
        )
        selected = await _command(
            runtime,
            "sh",
            "-c",
            'sed -n "$1" < "$2" | head -c "$3"',
            "read",
            f"{offset},{end}p;{end}q",
            target,
            str(budget),
        )
        lines = (
            selected.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
        )
        return _numbered(lines, args, offset - 1)
    start = max(0, offset - 1 if offset > 0 else len(lines) + offset)
    return _numbered(lines[start : start + args["limit"]], args, start)


def _numbered(lines, args, start):
    if lines:
        lines[0] = lines[0][args["column_offset"] :]
    return "".join(f"{i}\t{line}" for i, line in enumerate(lines, start + 1))[
        : args["max_chars"]
    ]


async def file_tool(runtime: Runtime, operation: str, args: dict, cwd: str, rg: str):
    base = PurePosixPath(cwd)
    path = PurePosixPath(args.get("path", "."))
    target = str(base / path)
    if operation == "read":
        result = await _read_lines(runtime, target, args)
    elif operation == "write":
        data = args["content"].encode()
        if args["mode"] == "append":
            await _append(runtime, target, data)
        else:
            await runtime.write(target, data)
        result = f"Wrote {len(data)} bytes to {path}"
    elif operation == "edit":
        original = await read_text(runtime, target)
        crlf = "\r\n" in original and "\n" not in original.replace("\r\n", "")
        content = original.replace("\r\n", "\n") if crlf else original
        count = content.count(args["old"])
        if count == 0 or (count != 1 and not args["replace_all"]):
            raise ValueError(
                f"old string must match {'at least once' if args['replace_all'] else 'exactly once'} (found {count})"
            )
        updated = content.replace(
            args["old"], args["new"], -1 if args["replace_all"] else 1
        )
        await runtime.write(
            target, (updated.replace("\n", "\r\n") if crlf else updated).encode()
        )
        result = f"Edited {path}"
    elif operation == "glob":
        names = await _glob(runtime, target, args["pattern"])
        matches = sorted(str(path / name) for name in names)
        start, limit = args["offset"], args["limit"]
        result = "\n".join(matches[start : start + limit if limit else None])
    elif operation == "ls":
        result = "\n".join(
            sorted(await _names(runtime, _LIST, target))[: args["limit"]]
        )
    elif operation == "grep":
        result = await remote_search.search(runtime, rg, cwd, args)
    elif operation == "directory":
        result = (
            await _command(
                runtime,
                "sh",
                "-c",
                'CDPATH= cd -- "$1" && pwd -P',
                "cd",
                str(base / (args["cd"] or ".")),
            )
        ).removesuffix("\n")
    elif operation == "patch":
        result = await _patch(runtime, base, args["patch"])
    else:
        raise ValueError(f"unknown file operation: {operation}")
    return result[:100000] + (TRUNCATED if len(result) > 100000 else "")
