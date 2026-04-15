"""Core XML parser, extraction, normalization, format instructions."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from .fields import FieldSpec

_log = logging.getLogger(__name__)

_XML_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)

def _coerce_bool(v: str) -> bool | None:
    normed = v.strip().lower()
    if normed in ("true", "yes", "1"):
        return True
    if normed in ("false", "no", "0"):
        return False
    return None


_COERCE: dict[type, Callable[[str], Any]] = {
    str: lambda v: v,
    int: int,
    float: float,
    bool: _coerce_bool,
    list: lambda v: [item.strip() for item in v.split(",")],
}


def _coerce(value: str, target_type: type) -> Any:
    fn = _COERCE.get(target_type)
    if fn is None:
        _log.debug("No coercion for type %s", target_type.__name__)
        return None
    try:
        return fn(value)
    except (ValueError, TypeError) as exc:
        _log.debug("Field coerce to %s failed: %s", target_type.__name__, exc)
        return None


def parse(text: str, schema: dict[str, type]) -> dict[str, Any] | None:
    xml_matches = _XML_TAG_RE.findall(text)
    if not xml_matches:
        return None
    result: dict[str, Any] = {}
    seen: set[str] = set()
    for tag, content in xml_matches:
        if tag not in schema:
            continue
        if tag in seen:
            raise ValueError(
                f"Duplicate XML tag '{tag}' in response — ambiguous commit"
            )
        seen.add(tag)
        coerced = _coerce(content.strip(), schema[tag])
        if coerced is not None:
            result[tag] = coerced
    return result or None


def normalize_fields(raw: dict[str, Any], specs: dict[str, FieldSpec]) -> dict[str, Any]:
    result = dict(raw)
    for key, value in raw.items():
        spec = specs.get(key)
        if spec is not None and spec.normalizer is not None:
            result[key] = spec.normalizer(value)
    return result


def extract_fields(text: str, specs: dict[str, FieldSpec]) -> dict[str, Any] | None:
    type_map = {k: v.type for k, v in specs.items()}
    raw = parse(text, type_map)
    if raw is None:
        return None
    return normalize_fields(raw, specs)


def generate_format_instructions(fields: Mapping[str, FieldSpec]) -> str:
    from .fields import BinaryScoring, EnumScoring, NumericScoring

    lines = ["After your reasoning, you MUST include the following XML tags at the end of your response:"]
    for name, spec in fields.items():
        scoring = spec.scoring
        if isinstance(scoring, BinaryScoring):
            lines.append(f"<{name}>{scoring.true_value} or {scoring.false_value}</{name}>")
        elif isinstance(scoring, EnumScoring):
            vals = ", ".join(scoring.values)
            lines.append(f"<{name}> YOUR {name.upper()} </{name}>  (exactly one of: {vals})")
        elif isinstance(scoring, NumericScoring):
            lines.append(f"<{name}>number between {scoring.min_val} and {scoring.max_val}</{name}>")
        elif spec.description:
            lines.append(f"<{name}>{spec.description}</{name}>")
        else:
            lines.append(f"<{name}>your {name} here ({spec.type.__name__})</{name}>")
    return "\n".join(lines)
