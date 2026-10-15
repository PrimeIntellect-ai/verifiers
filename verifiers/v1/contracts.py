"""Contracts: what a role's reply must be, and the repair turns that get it there.

A `Contract` turns a reply's text into a value with its `parse` function, which raises
when the reply breaks the contract. Any callable is a contract (a tag format, a line
syntax); `Contract.json(Model)` and `Contract.markdown(...)` are two stock ones:

    contract = vf.Contract.json(Verdict)                       # one JSON object
    contract = vf.Contract.markdown(                            # a markdown document
        frontmatter=DesignMeta,                                 # its `---` block, validated
        headings=["Task", "Request", "Crux", "Checks", "Build"],  # required `## ` sections
        items={"Checks": parse_check},                          # `- ` lines, each parsed
    )
    reply = await agent.until_valid(task, contract, attempts=3)
    reply.value, reply.trace

`Agent.until_valid` runs the agent, parses its reply, and while the reply breaks the
contract continues the same conversation with a repair turn: the `repair` prompt file
with the errors inserted (`{errors}`, `{attempt}`, `{attempts}`). Errors are one line
each, by line (`line 12: ...`) or by field (`checks.0.skill: ...`). After `attempts`
replies it raises `ContractFailed`, a role failure, never an outage. A run that fails
(its trace not ok) raises `RunFailed` with the trace's error, which is an `Outage` when
the error is one (`verifiers.v1.errors`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from verifiers.v1.trace import Error

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.trace import Trace

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

REPAIR_PROMPT = Path(__file__).resolve().parent / "repair.txt"
"""The default repair prompt file."""
MAX_ERRORS = 20
"""Error lines a repair turn shows at most."""


class ContractError(ValueError):
    """A reply that breaks its contract: `errors`, one line each."""

    def __init__(self, *errors: str) -> None:
        self.errors = [line for line in errors if line] or [
            "the reply breaks its contract"
        ]
        super().__init__("; ".join(self.errors))

    @classmethod
    def of(cls, error: BaseException, *, at: str = "") -> ContractError:
        """`error` as contract errors: a pydantic error by field (`at.loc: msg`), a
        `ContractError` as it is, anything else its message."""
        if isinstance(error, ContractError):
            return error if not at else cls(*(f"{at}: {line}" for line in error.errors))
        if isinstance(error, ValidationError):
            lines = []
            for item in error.errors(include_url=False):
                where = ".".join(
                    str(part) for part in (at, *item.get("loc", ())) if part != ""
                )
                lines.append(f"{where or 'value'}: {item.get('msg')}")
            return cls(*lines)
        return cls(f"{at}: {error}" if at else str(error) or type(error).__name__)


class ContractFailed(Exception):
    """A reply broke its contract on every attempt: a role failure (the model did not
    comply), not an infrastructure outage."""

    def __init__(
        self, errors: list[str], attempts: int, trace: Trace, *, seat: str | None = None
    ) -> None:
        self.errors, self.attempts, self.trace, self.seat = (
            errors,
            attempts,
            trace,
            seat,
        )
        self.trace_id = trace.id
        self.error = Error(
            type="ContractFailed", message="; ".join(errors)[:2000], seat=seat
        )
        super().__init__(
            f"the reply broke its contract after {attempts} attempts: {'; '.join(errors)[:1000]}"
        )


class RunFailed(Exception):
    """An agent run ended failed (`trace.ok` is False) where its reply was needed:
    `error` is its trace's error record, so `isinstance(e, Outage)` holds when the
    failure was infrastructure."""

    def __init__(self, trace: Trace, *, seat: str | None = None) -> None:
        self.trace, self.trace_id, self.seat = trace, trace.id, seat
        error = trace.last_error or Error(type="RolloutError", message="the run failed")
        self.error = (
            error.model_copy(update={"seat": seat}) if seat is not None else error
        )
        super().__init__(f"{self.error.type}: {self.error.message}")


@dataclass(frozen=True)
class Reply(Generic[T]):
    """A reply that met its contract: its parsed `value`, the trace that gave it, and
    every trace of the exchange (the first reply and each repair), oldest first."""

    value: T
    trace: Trace
    traces: list[Trace] = field(default_factory=list)

    @property
    def attempts(self) -> int:
        return len(self.traces)


@dataclass(frozen=True)
class Contract(Generic[T]):
    """What a reply must be: `parse(text)` returns its value or raises (a
    `ContractError` with line or field errors, a pydantic `ValidationError`, or any
    error whose message says what is wrong). `reply(trace, runtime)` reads the text
    from a finished run (default: `trace.last_reply`); one that reads a box file must
    read what the run left, since a kept run is not run again."""

    parse: Callable[[str], T]
    reply: Callable[[Trace, Runtime | None], Awaitable[str]] | None = None

    def check(self, text: str) -> T:
        """`parse(text)`, its failure raised as a `ContractError`."""
        try:
            return self.parse(text)
        except Exception as error:
            raise ContractError.of(error) from error

    async def text(self, trace: Trace, runtime: Runtime | None = None) -> str:
        return (
            await self.reply(trace, runtime)
            if self.reply is not None
            else trace.last_reply
        )

    @classmethod
    def json(cls, model: type[M] | Any, *, key: str | None = None) -> Contract[Any]:
        """One JSON object in the reply (`json_object`), validated as `model` (a pydantic
        model, or any type pydantic validates)."""
        from pydantic import TypeAdapter

        adapter = (
            None
            if isinstance(model, type) and issubclass(model, BaseModel)
            else TypeAdapter(model)
        )

        def parse(text: str) -> Any:
            found = json_object(text, key=key)
            return (
                adapter.validate_python(found)
                if adapter is not None
                else model.model_validate(found)
            )

        return cls(parse)

    @classmethod
    def markdown(
        cls,
        *,
        headings: Sequence[str],
        frontmatter: type[BaseModel] | None = None,
        items: Mapping[str, Callable[[str], Any]] | None = None,
        check: Callable[[Markdown], None] | None = None,
    ) -> Contract[Markdown]:
        """A markdown document (`parse_markdown`); `check(doc)` raises for what spans the
        document (e.g. at least one critical check)."""

        def parse(text: str) -> Markdown:
            doc = parse_markdown(
                text, headings=headings, frontmatter=frontmatter, items=items
            )
            if check is not None:
                check(doc)
            return doc

        return cls(parse)


# -- JSON -------------------------------------------------------------------------------


def json_object(text: str, *, key: str | None = None) -> dict[str, Any]:
    """The JSON object in a reply: the whole reply (or its one fenced block) when it is
    one, else the first balanced object in it; with `key`, the first object that has
    it (passing over an echoed example that does not parse). Raises `ContractError`."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```[\w-]*\n(.*?)\n?```", stripped, re.DOTALL)
    whole = fenced.group(1).strip() if fenced else stripped
    try:
        value = json.loads(whole)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and (key is None or key in value):
        return value
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and (key is None or key in value):
            return value
        start = stripped.find("{", start + 1)
    raise ContractError(
        f"the reply holds no JSON object{f' with a {key!r} key' if key else ''}"
    )


# -- markdown ---------------------------------------------------------------------------


class Markdown(BaseModel):
    """A parsed markdown reply: its frontmatter (the model, or a dict without one),
    each `## ` section's text, and the parsed `- ` lines of the item sections."""

    model_config = {"arbitrary_types_allowed": True}

    frontmatter: Any = None
    sections: dict[str, str] = {}
    items: dict[str, list[Any]] = {}
    lines: dict[str, int] = {}
    """Each section's heading line (1-based), for errors a `check` reports."""


_HEADING = re.compile(r"^##\s+(.+?)\s*#*\s*$")


def parse_markdown(
    text: str,
    *,
    headings: Sequence[str],
    frontmatter: type[BaseModel] | None = None,
    items: Mapping[str, Callable[[str], Any]] | None = None,
) -> Markdown:
    """Parse a markdown reply: an optional `---` frontmatter block (flat YAML: scalars,
    `[a, b]` lists and `- ` lists; validated as `frontmatter` when given, which makes
    it required), `## ` sections (every name in `headings` required, once), and in the
    `items` sections one `- ` line per item, each parsed by its function (blank lines
    and lines indented under an item are skipped). Every problem is reported at once,
    by line, as a `ContractError`."""
    lines = text.strip("\n").splitlines()
    errors: list[str] = []
    meta: dict[str, Any] | None = None
    body = 0
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
        if end is None:
            errors.append("line 1: the frontmatter opened by '---' is never closed")
        else:
            meta = _frontmatter(lines[1:end], first=2, errors=errors)
            body = end + 1
    found = Markdown()
    if frontmatter is not None:
        if meta is None and not any("frontmatter" in line for line in errors):
            errors.append("line 1: the reply must start with a '---' frontmatter block")
        elif meta is not None:
            try:
                found.frontmatter = frontmatter.model_validate(meta)
            except ValidationError as invalid:
                errors.extend(ContractError.of(invalid, at="frontmatter").errors)
    else:
        found.frontmatter = meta
    current: str | None = None
    chunks: dict[str, list[tuple[int, str]]] = {}
    for number, line in enumerate(lines[body:], start=body + 1):
        heading = _HEADING.match(line)
        if heading:
            current = heading.group(1).strip()
            if current in chunks:
                errors.append(f"line {number}: '## {current}' appears twice")
            found.lines[current] = number
            chunks[current] = []
        elif current is not None:
            chunks[current].append((number, line))
    for name in headings:
        if name not in chunks:
            errors.append(f"missing section '## {name}'")
    for name, chunk in chunks.items():
        found.sections[name] = "\n".join(line for _, line in chunk).strip()
    for name, parse in (items or {}).items():
        if name not in chunks:
            continue
        parsed: list[Any] = []
        for number, line in chunks[name]:
            if not line.strip() or line[:1].isspace():
                continue
            if not line.startswith(("- ", "* ")):
                errors.append(
                    f"line {number}: '## {name}' takes one '- ' item per line, not {line.strip()[:80]!r}"
                )
                continue
            try:
                parsed.append(parse(line[2:].strip()))
            except Exception as error:  # noqa: BLE001 - a bad item is the reply's
                errors.extend(
                    f"line {number}: {message}"
                    for message in ContractError.of(error).errors
                )
        if not parsed and not any(f"'## {name}'" in e for e in errors):
            errors.append(f"line {found.lines[name]}: '## {name}' lists no items")
        found.items[name] = parsed
    if errors:
        raise ContractError(*errors)
    return found


def _frontmatter(lines: list[str], *, first: int, errors: list[str]) -> dict[str, Any]:
    """Flat YAML: `key: value`, `key: [a, b]`, and `key:` followed by `- item` lines."""
    meta: dict[str, Any] = {}
    last: str | None = None
    for number, line in enumerate(lines, start=first):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if last is None or not isinstance(meta.get(last), list):
                errors.append(f"line {number}: a '- ' item with no list key above it")
            else:
                meta[last].append(_scalar(stripped[2:]))
            continue
        key, sep, value = stripped.partition(":")
        if not sep or not key.strip() or line[:1].isspace():
            errors.append(
                f"line {number}: expected 'key: value', not {stripped[:80]!r}"
            )
            continue
        last, value = key.strip(), value.strip()
        if not value:
            meta[last] = []
        elif value.startswith("[") and value.endswith("]"):
            meta[last] = [
                _scalar(part) for part in value[1:-1].split(",") if part.strip()
            ]
        else:
            meta[last] = _scalar(value)
    return meta


def _scalar(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


# -- repair -----------------------------------------------------------------------------


def repair_note(
    repair: Path | str | None, errors: list[str], attempt: int, attempts: int
) -> str:
    """The repair turn: the `repair` prompt file (default `REPAIR_PROMPT`) with
    `{errors}` (one `- ` line each, at most `MAX_ERRORS`), `{attempt}` (the reply just
    made) and `{attempts}` filled in; other braces stay as written."""
    template = Path(repair or REPAIR_PROMPT).read_text(encoding="utf-8")
    shown = errors[:MAX_ERRORS] + (
        [f"... and {len(errors) - MAX_ERRORS} more"] if len(errors) > MAX_ERRORS else []
    )
    fields = {
        "errors": "\n".join(f"- {line}" for line in shown),
        "attempt": str(attempt),
        "attempts": str(attempts),
    }
    return re.sub(
        r"\{(errors|attempt|attempts)\}", lambda match: fields[match.group(1)], template
    )


def repair_key(key: str | None, attempt: int) -> str | None:
    """The key of reply `attempt` (1-based) of a contract call keyed `key`: `key`, then
    `key.repair1`, `key.repair2`, ..."""
    if key is None:
        return None
    return key if attempt == 1 else f"{key}.repair{attempt - 1}"


__all__ = [
    "REPAIR_PROMPT",
    "Contract",
    "ContractError",
    "ContractFailed",
    "Markdown",
    "Reply",
    "RunFailed",
    "json_object",
    "parse_markdown",
    "repair_key",
    "repair_note",
]
