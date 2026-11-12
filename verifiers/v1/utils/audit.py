"""Bounded source evidence and redacted records for rubric reviews."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from verifiers.v1.trace import RECORD_DUMP_EXCLUDE, Trace

SOURCE_FILE_BYTES = 256 * 1024
SOURCE_TOTAL_BYTES = 2 * 1024 * 1024
SOURCE_ENTRIES = 512
SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|"
    r"authorization|credential|token)(?:$|[_-])",
    re.IGNORECASE,
)
PRIVATE_MAPS = {"env", "verifier_env", "environment", "headers", "extra_headers"}


def read_regular(path: Path, limit: int) -> bytes:
    """Read a bounded regular file, never following its final symlink."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"not a regular file: {path}")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"file exceeds {limit} bytes: {path}")
    return data


def source_evidence(directory: Path) -> dict:
    """Read task source without importing it or following links in the tree."""
    if directory.is_symlink():
        raise ValueError(f"source directory must not be a symlink: {directory}")
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"source is not a directory: {directory}")
    files, skipped = [], []
    used = visited = 0
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", "audit"}
    suffixes = {".py", ".sh", ".toml", ".json", ".yaml", ".yml", ".md", ".txt"}
    for parent, dirs, names, descriptor in os.fwalk(root, follow_symlinks=False):
        visited += 1
        if visited > SOURCE_ENTRIES:
            skipped.append({"path": "*", "reason": "source entry limit reached"})
            break
        kept = []
        for name in sorted(dirs):
            relative = str((Path(parent) / name).relative_to(root))
            if name in ignored:
                skipped.append({"path": relative, "reason": "excluded directory"})
            elif stat.S_ISDIR(
                os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode
            ):
                kept.append(name)
            else:
                skipped.append({"path": relative, "reason": "symlink directory"})
        dirs[:] = kept
        for name in sorted(names):
            visited += 1
            if visited > SOURCE_ENTRIES:
                skipped.append({"path": "*", "reason": "source entry limit reached"})
                return {"files": files, "skipped": skipped}
            relative = str((Path(parent) / name).relative_to(root))
            if name.startswith(".env") or (
                Path(name).suffix not in suffixes and name != "Dockerfile"
            ):
                skipped.append({"path": relative, "reason": "not task source"})
                continue
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=descriptor,
                )
                with os.fdopen(fd, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ValueError("not a regular file")
                    data = stream.read(
                        min(SOURCE_FILE_BYTES, SOURCE_TOTAL_BYTES - used) + 1
                    )
                if (
                    len(data) > SOURCE_FILE_BYTES
                    or used + len(data) > SOURCE_TOTAL_BYTES
                ):
                    raise ValueError("source byte limit reached")
                content = data.decode("utf-8")
                if "\x00" in content:
                    raise ValueError("binary content")
                used += len(data)
                files.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "content": content,
                    }
                )
            except (OSError, ValueError) as error:
                skipped.append({"path": relative, "reason": str(error)})
    return {"files": files, "skipped": skipped}


def redact(value):
    """Omit configuration credentials and remove their echoes elsewhere in a report."""
    secrets: set[str] = set()

    def sensitive(key: str) -> bool:
        return bool(SECRET_KEY.search(key)) and not key.endswith("_var")

    def strings(item):
        if isinstance(item, str) and len(item) >= 4:
            secrets.add(item)
        elif isinstance(item, dict):
            for child in item.values():
                strings(child)
        elif isinstance(item, list):
            for child in item:
                strings(child)

    def discover(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in PRIVATE_MAPS or sensitive(key):
                    strings(child)
                else:
                    discover(child)
        elif isinstance(item, list):
            for child in item:
                discover(child)

    discover(value)
    replacements = sorted(secrets, key=len, reverse=True)

    def url(match):
        try:
            parts = urlsplit(match[0])
            query = [
                (key, "[redacted]" if sensitive(key) else val)
                for key, val in parse_qsl(parts.query, keep_blank_values=True)
            ]
            return urlunsplit(
                (
                    parts.scheme,
                    parts.netloc.rsplit("@", 1)[-1],
                    parts.path,
                    urlencode(query),
                    parts.fragment,
                )
            )
        except ValueError:
            return "[redacted URL]"

    def scrub(item):
        if isinstance(item, dict):
            return {
                key: (
                    {name: "[redacted]" for name in child}
                    if key in PRIVATE_MAPS and isinstance(child, dict)
                    else "[redacted]"
                    if sensitive(key)
                    else scrub(child)
                )
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if isinstance(item, str):
            for secret in replacements:
                item = item.replace(secret, "[redacted]")
            item = re.sub(r"https?://[^\s<>\"']+", url, item)
            return re.sub(
                r"\bBearer\s+[^\s\"']+", "Bearer [redacted]", item, flags=re.IGNORECASE
            )
        return item

    return scrub(value)


def trace_record(trace: Trace, episode_id: str) -> dict:
    """Keep the full message graph and recorded config, omitting large token streams."""
    record = trace.model_dump(mode="json", exclude=RECORD_DUMP_EXCLUDE)
    return {
        "episode_id": episode_id,
        "model": trace.agent.config.model
        or next((call.model for call in trace.calls if call.model), "unknown"),
        "harness": trace.agent.config.harness.id
        if trace.agent.config.harness
        else "unknown",
        "runtime": trace.agent.config.runtime.type,
        "trace": record,
    }
