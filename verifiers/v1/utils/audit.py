"""Bounded source evidence and redacted records for rubric reviews."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import re
import stat
import tokenize
import tomllib
from collections.abc import Iterator
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
    r'''(?P<value>(?i:fr|rf|br|rb|r|u|f|b)?(?:"""(?:\\.|(?!""")[^\\])*"""|'''
    r"'''(?:\\.|(?!''')[^\\])*'''|"
    r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|[([{]|[^\s,;#}\]"']+)""",
    re.DOTALL,
)
F_STRING = re.compile(r"""(?i)(?:fr|rf|f)(?=["'])""")
URL = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"']+")
AUTHORIZATION = re.compile(r"\b(?:Bearer|Basic)\s+([^\s\"']+)", re.IGNORECASE)
AUTH_HEADER = re.compile(
    r"(?P<prefix>\b(?:Proxy-)?Authorization[ \t]*:[ \t]*)(?P<value>[^\r\n]+)",
    re.IGNORECASE,
)
LEGACY_ENV = re.compile(
    r"^[ \t]*(?i:ENV)[ \t]+(?P<key>[A-Za-z_][\w.-]*)[ \t]+"
    r"(?P<value>(?![ \t]*=)(?:\\\r?\n|[^\r\n])+)",
    re.MULTILINE,
)


def _group_end(text: str, start: int) -> int | None:
    """Find the extent of an expression starting with a grouped value."""
    # A malformed group in a trace can otherwise make tokenization, AST parsing,
    # and echo discovery process the entire remainder of a large log.
    tail = text[start : start + SOURCE_FILE_BYTES]
    closing = {"(": ")", "[": "]", "{": "}"}
    stack = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(tail).readline):
            if token.type == tokenize.ERRORTOKEN and token.string.strip():
                break
            operator = token.string if token.type == tokenize.OP else None
            if not stack and (
                token.type in {tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.COMMENT}
                or operator in {",", ";", ")", "]", "}"}
            ):
                line, column = token.start
                preceding = tail.split("\n", line - 1)[:-1]
                end = start + sum(len(part) + 1 for part in preceding) + column
                return end if end < start + len(tail) or end == len(text) else None
            if operator in closing:
                stack.append(closing[operator])
            elif operator in {")", "]", "}"} and stack.pop() != operator:
                break
    except (tokenize.TokenError, SyntaxError):
        pass
    # An incomplete credential expression has no reliable value boundary.
    return None


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
        for segment in str(key).split("."):
            segment = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", segment).lower()
            if not segment.endswith("_var") and SECRET_KEY.search(segment):
                return True
        return False

    def strings(item):
        if isinstance(item, ast.JoinedStr):
            for part in item.values:
                # Literal separators and other short fragments of a template
                # say nothing about the complete interpolated credential.
                if (
                    isinstance(part, ast.Constant)
                    and len(part.value.strip()) >= 4
                    and any(char.isalnum() for char in part.value)
                ):
                    strings(part.value)
            return
        if isinstance(item, bytes):
            if item:
                secrets.add(repr(item)[2:-1])
            item = item.decode("utf-8", errors="backslashreplace")
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            item = str(item)
        if isinstance(item, str) and item:
            secrets.add(item)
            secrets.add(json.dumps(item, ensure_ascii=False)[1:-1])
        elif isinstance(item, dict):
            for child in item.values():
                strings(child)
        elif isinstance(item, (list, tuple, set)):
            for child in item:
                strings(child)

    def assignment_value(raw):
        try:
            expression = ast.parse(raw, mode="eval").body
            if isinstance(expression, ast.JoinedStr):
                # Collect literal text without evaluating formatted expressions.
                parts = [
                    part.value
                    for part in expression.values
                    if isinstance(part, ast.Constant)
                ]
                return (
                    expression
                    if len(parts) != len(expression.values)
                    else "".join(parts)
                )
            return ast.literal_eval(expression)
        except (ValueError, SyntaxError):
            # Shell quotes permit literal newlines that Python's single-quoted
            # literals reject. Their echoes omit quotes.
            if raw[:1] in {"'", '"'} and raw[-1:] == raw[:1]:
                return raw[1:-1]
            return raw

    def assignments(text: str) -> Iterator[tuple[re.Match[str], int | None]]:
        end = 0
        for match in ASSIGNMENT.finditer(text):
            if match.start() < end:
                continue
            start, end = match.span("value")
            if sensitive(match["key"]):
                if text[start] in "([{":
                    end = _group_end(text, start)
                elif F_STRING.match(text, start):
                    # A quote in a formatted expression can end the regex match early.
                    try:
                        ast.parse(match["value"], mode="eval")
                    except SyntaxError:
                        end = _group_end(text, start)
            yield match, end
            if end is None:
                break

    def group_values(raw: str, *, incomplete: bool = False) -> None:
        # A shell or JavaScript object is not necessarily a Python literal.
        # Gather the values of its inner fields independently of parsing the group.
        for match in ASSIGNMENT.finditer(raw):
            if match["value"] not in ("(", "[", "{") and (
                not incomplete or sensitive(match["key"])
            ):
                strings(assignment_value(match["value"]))

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
            for directive in LEGACY_ENV.finditer(item):
                if sensitive(directive["key"]):
                    raw = re.sub(r"\\\r?\n", "", directive["value"]).strip()
                    strings(assignment_value(raw))
            for header in AUTH_HEADER.finditer(item):
                strings(header["value"])
                for param in ASSIGNMENT.finditer(header["value"]):
                    if param["key"].lower() == "response":
                        strings(assignment_value(param["value"]))
            for match, end in assignments(item):
                start = match.start("value")
                if end is None:
                    group_values(
                        item[start : start + SOURCE_FILE_BYTES], incomplete=True
                    )
                    break
                raw = item[start:end]
                decoded = assignment_value(raw)
                if sensitive(match["key"]):
                    if decoded == raw and raw.startswith(("(", "[", "{")):
                        group_values(raw)
                    else:
                        strings(decoded)
                elif isinstance(decoded, ast.JoinedStr):
                    for part in decoded.values:
                        if isinstance(part, ast.Constant):
                            discover(part.value)
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
                        strings(parts.password)
                        strings(unquote(parts.password))
                    for key, child in parse_qsl(parts.query, keep_blank_values=True):
                        if sensitive(key):
                            strings(child)
                except ValueError:
                    continue

    discover(value)
    marker = r"(?P<marker>\[redacted(?: URL)?\]|%5Bredacted%5D)"
    replacements = sorted(secrets, key=len, reverse=True)
    patterns = [marker]
    for secret in replacements:
        escaped = re.escape(secret)
        if len(secret) < 4:
            boundary = (
                r"\w" if any(char.isalnum() for char in secret) else r"""[^\s"'\x60]"""
            )
            escaped = rf"(?<!{boundary}){escaped}(?!{boundary})"
        patterns.append(escaped)
    echoes = re.compile("|".join(patterns))

    def url(match):
        try:
            parts = urlsplit(match[0])
            original_query = parse_qsl(parts.query, keep_blank_values=True)
            query = [
                (key, "[redacted]" if sensitive(key) else val)
                for key, val in original_query
            ]
            if "@" not in parts.netloc and query == original_query:
                return match[0]
            return urlunsplit(
                (
                    parts.scheme,
                    parts.netloc.rsplit("@", 1)[-1],
                    parts.path,
                    parts.query if query == original_query else urlencode(query),
                    parts.fragment,
                )
            )
        except ValueError:
            return "[redacted URL]"

    def scrub_assignments(text: str) -> str:
        parts = []
        cursor = 0
        for match, end in assignments(text):
            start = match.start("value")
            if sensitive(match["key"]):
                # An annotation-shaped field may also be a credential-bearing
                # string in a log. Remove it here without treating its type as
                # an echoed secret in ordinary source elsewhere.
                separator = ":" if match["annotation"] else match["separator"]
                parts.extend(
                    (
                        text[cursor : match.start()],
                        match["target"],
                        separator,
                        ' "[redacted]"',
                    )
                )
            else:
                raw = text[start:end]
                decoded = assignment_value(raw)
                if isinstance(decoded, ast.JoinedStr):
                    changed = False
                    for part in decoded.values:
                        if isinstance(part, ast.Constant):
                            clean = scrub(part.value)
                            changed |= clean != part.value
                            part.value = clean
                    if changed:
                        raw = ast.unparse(decoded)
                elif isinstance(decoded, str) and decoded != raw:
                    cleaned = scrub(decoded)
                    if cleaned != decoded:
                        raw = json.dumps(cleaned, ensure_ascii=False)
                parts.extend((text[cursor:start], raw))
            cursor = len(text) if end is None else end
        parts.append(text[cursor:])
        return "".join(parts)

    def scrub_echoes(text: str) -> str:
        return echoes.sub(lambda match: match["marker"] or "[redacted]", text)

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
            item = LEGACY_ENV.sub(
                lambda match: (
                    item[match.start() : match.start("value")] + '"[redacted]"'
                    if sensitive(match["key"])
                    else match[0]
                ),
                item,
            )
            item = scrub_assignments(item)
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
