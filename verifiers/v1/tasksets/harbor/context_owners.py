"""Root ownership for the files an image copied from its build context.

Docker's `COPY` and `ADD` (without `--chown`) create every file root-owned. Prime's image
builder keeps the owner the files had in the uploaded context instead: the uid of whoever
registered the task (1000), so a grader that insists its private files are root's own
(`private verifier source permissions are too broad`), or a sandbox that relies on root
owning them, fails in the box and not under Docker.

`context_files` lists, from the task's Dockerfile, each file the final stage copied from
the context, where it landed and the owner the context gave it; `restore_owners` gives
those still carrying that owner back to root. Files a later `RUN` (or a context script
it runs) passes to `chown` are left alone, since the Dockerfile set their owner itself, and so is anything the context
did not hold (created later under the same uid).
"""

import asyncio
import os
import posixpath
import re
import shlex
from pathlib import Path

from verifiers.v1.runtimes import Runtime
from verifiers.v1.tasksets.harbor.entrypoint import exec_form, instructions

ARCHIVE = re.compile(r"\.(tar|tgz|tbz2?|txz|tar\.(gz|bz2|xz|zst))$")
RESTORE = r"""
while IFS=' ' read -r owner kind name path; do
  # A file copied onto a directory that exists in the image lands inside it.
  [ "$kind" = f ] && [ -d "$path" ] && path="$path/$name"
  [ "$(stat -c %u:%g "$path" 2>/dev/null)" = "$owner" ] && chown -h 0:0 "$path"
done < "$1"
rm -f "$1"
"""


def _sources(arguments: str) -> tuple[list[str], str, dict[str, str]] | None:
    """A COPY/ADD's sources, destination and flags; None when it cannot be read."""
    flags: dict[str, str] = {}
    words = exec_form(arguments)
    if words is None:
        try:
            words = shlex.split(arguments)
        except ValueError:
            return None
    while words and words[0].startswith("--"):
        name, _, value = words.pop(0)[2:].partition("=")
        flags[name] = value
    if len(words) < 2 or words[0].startswith("<<"):
        return None
    return words[:-1], words[-1], flags


def _chowned(command: str) -> list[str]:
    """Absolute paths a RUN passes to `chown` (a parent's `-R` covers its tree)."""
    paths = []
    for match in re.finditer(r"\bchown\b([^;&|\n]*)", command):
        for word in match[1].split():
            if word.startswith("/"):
                paths.append(posixpath.normpath(word.rstrip("/")) or "/")
    return paths


def context_files(dockerfile: Path) -> list[tuple[str, bool, int, int]]:
    """(path in the image, whether it is a file, uid, gid) of each regular file and
    directory the final stage copied from the build context without `--chown`, in the
    context's ownership. A lone file copied to a path without a trailing slash lands
    there, or inside it when the image has a directory there (`restore_owners`)."""
    if not dockerfile.is_file():
        return []
    context = dockerfile.parent
    steps = instructions(dockerfile.read_text(errors="replace"))
    final = max((i for i, (k, _) in enumerate(steps) if k == "FROM"), default=-1)
    workdir = "/"
    copies: list[tuple[int, str, str]] = []
    for index, (keyword, arguments) in enumerate(steps):
        if keyword == "WORKDIR":
            workdir = posixpath.normpath(posixpath.join(workdir, arguments.strip()))
        if index <= final or keyword not in ("COPY", "ADD"):
            continue
        parsed = _sources(arguments)
        if parsed is None:
            continue
        sources, destination, flags = parsed
        if "from" in flags or "chown" in flags or "$" in destination:
            continue
        target = posixpath.join(workdir, destination)
        many = len(sources) > 1 or destination.endswith("/")
        for source in sources:
            if keyword == "ADD" and (
                re.match(r"\w+://", source) or ARCHIVE.search(source)
            ):
                continue
            for path in (
                sorted(context.glob(source.lstrip("/")))
                if any(c in source for c in "*?[")
                else [context / source.lstrip("/")]
            ):
                if not path.exists():
                    continue
                if path.is_dir():
                    for item in [path, *sorted(path.rglob("*"))]:
                        if item.is_symlink() or not (item.is_file() or item.is_dir()):
                            continue
                        relative = item.relative_to(path).as_posix()
                        copies.append(
                            (index, str(item), posixpath.join(target, relative))
                        )
                elif path.is_file():
                    landed = posixpath.join(target, path.name) if many else target
                    copies.append((index, str(path), landed))
    # What a RUN executes includes the context scripts it calls (`RUN bash
    # /tmp/setup.sh`, where setup.sh hands /app to the agent): their text counts too.
    scripts = {
        landed: source
        for _, source, landed in copies
        if os.path.isfile(source) and os.path.getsize(source) < 1_000_000
    }

    def executed(arguments: str) -> str:
        text = [arguments]
        for landed, source in scripts.items():
            name = posixpath.basename(landed)
            if landed in arguments or (
                name in arguments
                and re.search(rf"(^|[\s/]){re.escape(name)}(\s|$|;|&)", arguments)
            ):
                text.append(Path(source).read_text(errors="replace"))
        return "\n".join(text)

    # What each RUN chowns, read once: a context of thousands of files would otherwise
    # rescan every script for every copied file, for hours.
    chowned = {
        index: _chowned(executed(arguments))
        for index, (keyword, arguments) in enumerate(steps)
        if keyword == "RUN"
    }
    owned = []
    for index, source, landed in copies:
        landed = posixpath.normpath(landed)
        later = [path for at, paths in chowned.items() if at > index for path in paths]
        if any(landed == p or landed.startswith(p.rstrip("/") + "/") for p in later):
            continue
        stat = os.stat(source)
        owned.append((landed, os.path.isfile(source), stat.st_uid, stat.st_gid))
    return [entry for entry in owned if entry[2:] != (0, 0)]


async def restore_owners(runtime: Runtime, dockerfile: Path) -> int:
    """Give the context's files in the box back to root (`context_files`); returns how
    many were listed."""
    # Off the event loop: reading a large context takes a while, and every other box
    # this process drives waits while it does.
    files = await asyncio.to_thread(context_files, dockerfile)
    if not files:
        return 0
    listing = "".join(
        f"{uid}:{gid} {'f' if is_file else 'd'} {posixpath.basename(path)} {path}\n"
        for path, is_file, uid, gid in files
        if "\n" not in path and " " not in posixpath.basename(path)
    )
    name = f"/tmp/vf-context-owners-{os.getpid()}-{id(files)}"
    await runtime.write(name, listing.encode())
    await runtime.run(["sh", "-c", RESTORE, "vf", name], {})
    return len(files)
