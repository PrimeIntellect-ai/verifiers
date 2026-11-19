"""Bounded source evidence and redacted records for rubric reviews."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import tomllib
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from verifiers.v1.trace import RECORD_DUMP_EXCLUDE, Trace
from verifiers.v1.utils.files import read_regular

SOURCE_FILE_BYTES = 256 * 1024
SOURCE_TOTAL_BYTES = 2 * 1024 * 1024
SOURCE_ENTRIES = 512
SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|"
    r"passwd|passphrase|private[_-]?key|authorization|credentials?|cookies?|token)(?:$|[_-])",
    re.IGNORECASE,
)
ASSIGNMENT = re.compile(
    r"""(?<![\w.-])(?P<target>["']?(?P<key>[A-Za-z_][\w.-]*)["']?[ \t]*\]?[ \t]*)"""
    r"""(?P<separator>:(?P<annotation>[ \t]*(?=[A-Za-z_"'])"""
    r"""(?:[\w.\[\], \t|]|"(?:\\.|[^"\\\r\n])*"|'(?:\\.|[^'\\\r\n])*')+?)=|[:=])\s*"""
    r'''(?P<value>(?:[rRuU])?(?:"""(?:\\.|(?!""")[^\\])*"""|'''
    r"'''(?:\\.|(?!''')[^\\])*'''|"
    r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|[^\s,;#}\]"']+)""",
    re.DOTALL,
)
URL = re.compile(r"https?://[^\s<>\"']+")
AUTHORIZATION = re.compile(r"\b(?:Bearer|Basic)\s+([^\s\"']+)", re.IGNORECASE)
AUTH_HEADER = re.compile(
    r"(?P<prefix>\b(?:Proxy-)?Authorization[ \t]*:[ \t]*)(?P<value>[^\r\n]+)",
    re.IGNORECASE,
)


def _source_configs(item):
    """Parse configuration evidence before discovering secrets; never emit failed parses."""
    if isinstance(item, list):
        return [_source_configs(child) for child in item]
    if not isinstance(item, dict):
        return item
    result = {key: _source_configs(child) for key, child in item.items()}
    path, content = result.get("path"), result.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return result
    suffix = Path(path).suffix.lower()
    if suffix not in {".toml", ".json", ".yaml", ".yml"}:
        return result
    try:
        if suffix == ".toml":
            parsed = tomllib.loads(content)
        elif suffix == ".json":
            parsed = json.loads(content)
        else:
            import yaml

            # Alias graphs can be cyclic or expand exponentially when represented
            # as JSON evidence. Omit these configs instead of unbounding review.
            if any(
                isinstance(token, yaml.tokens.AliasToken)
                for token in yaml.scan(content)
            ):
                raise ValueError("YAML aliases are not supported in review evidence")
            parsed = yaml.safe_load(content)
        if not isinstance(parsed, (dict, list)):
            raise TypeError("configuration must be a mapping or list")
        # Normalize TOML/YAML scalar types while refusing recursive structures.
        result["content"] = json.loads(json.dumps(parsed, default=str))
        result["content_format"] = "parsed configuration (comments omitted)"
    except Exception:  # noqa: BLE001 - config parsing must fail closed without leaking source
        # Parser diagnostics can quote the credential-bearing source. Neither
        # those diagnostics nor unparsed config may reach the external judge.
        result["content"] = "[omitted: configuration could not be parsed safely]"
    return result


def source_evidence(directory: Path) -> dict:
    """Read task source without importing it or following links in the tree."""
    if directory.is_symlink():
        raise ValueError(f"source directory must not be a symlink: {directory}")
    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"source is not a directory: {directory}")
    files, skipped = [], []
    used, visited, limited = 0, 1, False  # The root is an entry too.
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", "audit"}
    suffixes = {".py", ".sh", ".toml", ".json", ".yaml", ".yml", ".md", ".txt"}

    def visit(descriptor: int, prefix: Path) -> None:
        nonlocal used, visited, limited
        # Bound enumeration itself before sorting. fwalk/walk materialize a
        # whole directory, including ignored entries, before callers can limit it.
        names = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if visited >= SOURCE_ENTRIES:
                    limited = True
                    break
                visited += 1
                names.append(entry.name)
        for name in sorted(names):
            relative = str(prefix / name)
            try:
                mode = os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError("symlink")
                if stat.S_ISDIR(mode):
                    if name in ignored:
                        raise ValueError("excluded directory")
                    if visited >= SOURCE_ENTRIES:
                        limited = True
                        continue
                    child = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY,
                        dir_fd=descriptor,
                    )
                    try:
                        visit(child, prefix / name)
                    finally:
                        os.close(child)
                    continue
                if name.startswith(".env") or (
                    Path(name).suffix not in suffixes and name != "Dockerfile"
                ):
                    raise ValueError("not task source")
                data = read_regular(
                    name,
                    min(SOURCE_FILE_BYTES, SOURCE_TOTAL_BYTES - used),
                    dir_fd=descriptor,
                )
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

    descriptor = os.open(root, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
    try:
        visit(descriptor, Path())
    finally:
        os.close(descriptor)
    if limited:
        skipped.append({"path": "*", "reason": "source entry limit reached"})
    return {"files": files, "skipped": skipped}


def redact(value):
    """Omit configuration credentials and remove their echoes elsewhere in a report."""
    value = _source_configs(value)
    secrets: set[str] = set()

    def sensitive(key: str) -> bool:
        key = str(key).rsplit(".", 1)[-1]
        key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
        return bool(SECRET_KEY.search(key)) and not key.endswith("_var")

    def strings(item):
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            item = str(item)
        if isinstance(item, str) and len(item) >= 4:
            secrets.add(item)
            secrets.add(json.dumps(item, ensure_ascii=False)[1:-1])
        elif isinstance(item, dict):
            for child in item.values():
                strings(child)
        elif isinstance(item, list):
            for child in item:
                strings(child)

    def assignment_value(raw):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            # Shell quotes permit literal newlines that Python's single-quoted
            # literals reject. Their echoes omit quotes.
            if raw[:1] in {"'", '"'} and raw[-1:] == raw[:1]:
                return raw[1:-1]
            return raw

    def discover(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if sensitive(key):
                    strings(child)
                discover(child)
        elif isinstance(item, list):
            for child in item:
                discover(child)
        elif isinstance(item, str):
            for header in AUTH_HEADER.finditer(item):
                strings(header["value"])
                for param in ASSIGNMENT.finditer(header["value"]):
                    if param["key"].lower() == "response":
                        strings(assignment_value(param["value"]))
            for match in ASSIGNMENT.finditer(item):
                raw = match["value"]
                decoded = assignment_value(raw)
                if sensitive(match["key"]):
                    strings(decoded)
                    if annotation := match["annotation"]:
                        strings(annotation.strip())
                elif isinstance(decoded, str) and decoded != raw:
                    # A harmless outer key can hold commands or config text with
                    # its own credential assignments, including escaped quotes.
                    discover(decoded)
            for match in AUTHORIZATION.finditer(item):
                strings(match[1])
            for match in URL.finditer(item):
                try:
                    parts = urlsplit(match[0])
                    if parts.password:
                        strings(unquote(parts.password))
                    for key, child in parse_qsl(parts.query, keep_blank_values=True):
                        if sensitive(key):
                            strings(child)
                except ValueError:
                    continue

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

    def scrub_assignment(match):
        if sensitive(match["key"]):
            separator = ":" if match["annotation"] else match["separator"]
            return match["target"] + separator + ' "[redacted]"'
        raw = match["value"]
        decoded = assignment_value(raw)
        if isinstance(decoded, str) and decoded != raw:
            cleaned = scrub(decoded)
            if cleaned != decoded:
                prefix = match.string[match.start() : match.start("value")]
                return prefix + json.dumps(cleaned, ensure_ascii=False)
        return match[0]

    def scrub_echoes(text: str) -> str:
        for secret in replacements:
            text = text.replace(secret, "[redacted]")
        return text

    def scrub(item):
        if isinstance(item, dict):
            cleaned = {}
            suffixes: dict[str, int] = {}
            for key, child in item.items():
                safe = scrub_echoes(key) if isinstance(key, str) else scrub(key)
                unique = safe
                if unique in cleaned:
                    base = str(safe)
                    suffix = suffixes.get(base, 2)
                    while (unique := f"{base} [{suffix}]") in cleaned:
                        suffix += 1
                    suffixes[base] = suffix + 1
                cleaned[unique] = "[redacted]" if sensitive(key) else scrub(child)
            return cleaned
        if isinstance(item, list):
            return [scrub(child) for child in item]
        if isinstance(item, str):
            item = AUTH_HEADER.sub(lambda match: match["prefix"] + '"[redacted]"', item)
            item = ASSIGNMENT.sub(scrub_assignment, item)
            item = URL.sub(url, item)
            item = AUTHORIZATION.sub(
                lambda match: match[0].split()[0] + " [redacted]", item
            )
            return scrub_echoes(item)
        if isinstance(item, (int, float)) and str(item) in secrets:
            return "[redacted]"
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
