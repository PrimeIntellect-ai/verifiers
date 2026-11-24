"""Stage a native ripgrep when the worker sandbox lacks it, and stream its matches."""

import asyncio
import base64
import json
from pathlib import PurePosixPath

from verifiers.v1.harnesses.utils.install import (
    cached_npm_tarball_members,
    stage_executable,
)
from verifiers.v1.runtimes import Runtime

_VERSION = "1.18.0"  # @vscode/ripgrep bundles static Linux binaries and PCRE2.
_INTEGRITY = {
    "linux-x64": "mQ3bVrUpnD2vs7QT0vX90Lt0cnUq467uFtEktIdsJJmW296RoSULRGqWgzG1AKxyBpNDD6l4ZO4qKf6SgyC23Q==",
    "linux-arm64": "lQ/5zTG++U0E3IhVgS4EPTTn/U4okncaRMM5GOFfOYZywS4nuD31GhkHbNYlDk5CuDC68+hYJ0/eQeyCKJDA+g==",
    "darwin-arm64": "r3ktHSvbFycQNF6sl7sNDPocpsI7J+mEzh1IaZFkY0spm3k2Z9t8hPAeOK7+p0l6p6/swkQC14XWX01low+94Q==",
    "darwin-x64": "25b4gWbL138dGuQU244ebCKKc0q05ULBMoFSz9oAEUHNeqK/lOJViDS7DRvbDazzAzSEdan391Znks/R5mkaTQ==",
}


async def _binary(platform: str) -> bytes:
    name = f"ripgrep-{platform}"
    url = f"https://registry.npmjs.org/@vscode/{name}/-/{name}-{_VERSION}.tgz"
    files = await cached_npm_tarball_members(
        url,
        lambda member: "rg" if member == "package/bin/rg" else None,
        sha512_b64=_INTEGRITY[platform],
    )
    return files["rg"]


def ripgrep_path(runtime: Runtime) -> str:
    return runtime.temp_path("vf-tools", f"ripgrep-{_VERSION}")


async def prepare_ripgrep(runtime: Runtime, target: str) -> None:
    key = f"ripgrep:{target}:{runtime.user or ''}"
    async with runtime.backend.cache.setup_lock:
        if key not in runtime.backend.cache.installed:
            await _stage(runtime, target)
            runtime.backend.cache.installed.add(key)


async def _stage(runtime: Runtime, target: str) -> None:
    probe = await runtime.run(
        [
            "sh",
            "-c",
            (
                "if rg --pcre2-version >/dev/null 2>&1; then command -v rg; "
                'elif [ -x "$1" ]; then printf "%s\\n" "$1"; else uname -s; uname -m; fi'
            ),
            "ripgrep",
            target,
        ],
        {},
    )
    info = probe.stdout.strip().splitlines()
    if probe.exit_code or len(info) not in {1, 2}:
        raise ValueError(
            f"could not locate worker ripgrep or architecture: {probe.stderr.strip()}"
        )
    if len(info) == 1:
        return
    system, arch = info
    arch = {"x86_64": "x64", "aarch64": "arm64"}.get(arch, arch)
    platform = f"{system.lower()}-{arch}"
    if platform not in _INTEGRITY:
        raise ValueError(
            f"no native ripgrep for {platform}; install rg with PCRE2 in the image"
        )
    await stage_executable(runtime, target, await _binary(platform))


async def _records(stream, separators):
    buffered = bytearray()
    fields = []
    received = 0
    async for chunk in stream:
        received += len(chunk)
        if received > 16 * 1024 * 1024:
            raise ValueError("search output exceeds 16 MiB; narrow the path or pattern")
        buffered.extend(chunk)
        while (at := buffered.find(separators[len(fields)])) >= 0:
            fields.append(bytes(buffered[:at]))
            del buffered[: at + 1]
            if len(fields) == len(separators):
                yield fields
                fields = []
        if len(buffered) > 4 * 1024 * 1024:
            raise ValueError("matching line exceeds 4 MiB; narrow the search")
    if buffered or fields:
        raise ValueError("incomplete ripgrep response")


def _value(field):
    if "text" in field:
        return field["text"]
    return base64.b64decode(field["bytes"]).decode(errors="replace")


async def search(runtime: Runtime, binary: str, cwd: str, args: dict) -> str:
    mode = args["output_mode"]
    command = [
        binary,
        "--no-config",
        "--no-messages",
        "--no-ignore",
        "--hidden",
        "--sort=path",
        "--engine=auto",
        "--color=never",
    ]
    if args["glob"]:
        command.extend(["--glob", args["glob"]])
    command.extend(["--glob", "!.git", "--glob", "!**/.git/**"])
    if args["ignore_case"]:
        command.append("--ignore-case")
    if args["literal"]:
        command.append("--fixed-strings")
    if args["multiline"]:
        command.extend(["--multiline", "--multiline-dotall"])
    if mode == "files_with_matches":
        command.extend(["--files-with-matches", "--null"])
        separators = (b"\0",)
    elif mode in {"count", "count_matches"}:
        command.extend(["--count-matches", "--with-filename", "--null"])
        separators = (b"\0", b"\n")
    else:
        before, after = (
            max(args["before"], args["context"]),
            max(args["after"], args["context"]),
        )
        command.extend(["--json", "-B", str(before), "-A", str(after)])
        separators = (b"\n",)
    command.extend(
        ["-e", args["pattern"], "--", "./-" if args["path"] == "-" else args["path"]]
    )
    process = await runtime.exec(
        [
            "sh",
            "-c",
            'CDPATH= cd -- "$1" || exit 1; shift; binary=$1; shift; [ -x "$binary" ] || binary=rg; exec "$binary" "$@"',
            "search",
            cwd or ".",
            *command,
        ],
        {},
    )

    async def errors():
        tail = b""
        async for chunk in process.stderr:
            tail = (tail + chunk)[-1024:]
        return tail.decode(errors="replace").strip()

    stderr = asyncio.create_task(errors())
    result = []
    skipped = size = 0
    try:
        async with asyncio.timeout(60):
            async for fields in _records(process.stdout, separators):
                if mode == "content":
                    event = json.loads(fields[0])
                    if event["type"] not in {"match", "context"}:
                        continue
                    data = event["data"]
                    path = PurePosixPath(_value(data["path"]))
                    matches = (
                        f"{path}:{i}:{line}"
                        for i, line in enumerate(
                            _value(data["lines"]).splitlines(), data["line_number"]
                        )
                    )
                else:
                    path = PurePosixPath(fields[0].decode(errors="replace"))
                    matches = [
                        str(path) if len(fields) == 1 else f"{path}:{int(fields[1])}"
                    ]
                for line in matches:
                    if skipped < args["offset"]:
                        skipped += 1
                        continue
                    result.append(line)
                    size += len(line) + 1
                    if size > 100000 or (
                        args["limit"] and len(result) >= args["limit"]
                    ):
                        return "\n".join(result)
            status = await process.wait()
            if status not in (0, 1):
                reason = await stderr
                # --no-messages silences inaccessible entries; a malformed pattern still has diagnostics.
                if status != 2 or reason:
                    raise ValueError(reason or f"ripgrep exited with status {status}")
        return "\n".join(result)
    except TimeoutError as error:
        raise ValueError("workspace search timed out after 60s") from error
    finally:
        if await process.poll() is None:
            await process.kill()
        stderr.cancel()
        await asyncio.gather(stderr, return_exceptions=True)
