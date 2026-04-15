"""Field-level scoring: mode types, classifiers, normalizers, and field resolution."""

from __future__ import annotations

import contextlib
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Required, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# FieldSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    type: type
    description: str = ""
    scoring: ScoringMode | None = None
    normalizer: Callable[[Any], Any] | None = None


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------


def _canon_base(s: str) -> str:
    """Casefold, normalize separators, collapse whitespace."""
    s = s.casefold()
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


_BINARY_STRIP_CHARS = ".,!?;:'\"()[]{}«»\u201c\u201d\u2018\u2019\u3002\uff1f\uff01"


def _canon_binary(s: str) -> str:
    """Base canon + strip boundary punctuation."""
    return _canon_base(s).strip(_BINARY_STRIP_CHARS)


def _canon_enum(s: str) -> str:
    """Base canon only (preserves trailing +, etc.)."""
    return _canon_base(s)


# ---------------------------------------------------------------------------
# Classification results (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinaryClassification:
    """Result of classifying a value as binary true/false."""

    is_true: bool | None
    is_valid: bool
    canonical: str


@dataclass(frozen=True)
class EnumClassification:
    """Result of classifying a value against an enum set."""

    canonical: str | None
    is_valid: bool


@dataclass(frozen=True)
class NumericClassification:
    """Result of classifying a numeric value within bounds."""

    value: float | None
    is_valid: bool


# ---------------------------------------------------------------------------
# Classifiers (single source of truth for normalizers + metrics)
# ---------------------------------------------------------------------------


def classify_binary(value: Any, true_value: str, false_value: str) -> BinaryClassification:
    """Classify value as true/false/invalid using exact match + first-token fallback."""
    s = _canon_binary(str(value))
    tv = _canon_binary(true_value)
    fv = _canon_binary(false_value)

    # Exact match
    if s == tv:
        return BinaryClassification(is_true=True, is_valid=True, canonical=true_value)
    if s == fv:
        return BinaryClassification(is_true=False, is_valid=True, canonical=false_value)

    # First-token fallback (strip punct from token too)
    first = s.split()[0].strip(_BINARY_STRIP_CHARS) if s else ""
    if first == tv:
        return BinaryClassification(is_true=True, is_valid=True, canonical=true_value)
    if first == fv:
        return BinaryClassification(is_true=False, is_valid=True, canonical=false_value)

    return BinaryClassification(is_true=None, is_valid=False, canonical=str(value))


def classify_enum(value: Any, values: tuple[str, ...]) -> EnumClassification:
    """Classify value against enum set. Exact match first, then first-token fallback."""
    canon_map = {_canon_enum(v): v for v in values}
    s = _canon_enum(str(value))
    if s in canon_map:
        return EnumClassification(canonical=canon_map[s], is_valid=True)
    # First-token fallback: "A, B, tie" -> try "A"; "Expert A wins" -> try "expert"
    first = s.split()[0].strip(_BINARY_STRIP_CHARS) if s else ""
    if first in canon_map:
        return EnumClassification(canonical=canon_map[first], is_valid=True)
    return EnumClassification(canonical=None, is_valid=False)


def classify_numeric(value: Any, min_val: float, max_val: float) -> NumericClassification:
    """Classify numeric value: isinstance primary path, defensive float() fallback."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and min_val <= value <= max_val:
            return NumericClassification(value=float(value), is_valid=True)
        return NumericClassification(value=None, is_valid=False)
    # Defensive: try float() for string/replay inputs
    if isinstance(value, bool):
        return NumericClassification(value=None, is_valid=False)
    with contextlib.suppress(ValueError, TypeError):
        f = float(value)
        if math.isfinite(f) and min_val <= f <= max_val:
            return NumericClassification(value=f, is_valid=True)
        return NumericClassification(value=None, is_valid=False)
    return NumericClassification(value=None, is_valid=False)


# ---------------------------------------------------------------------------
# Scoring mode types (frozen dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoringMode:
    """Base scoring mode."""

    mode: str = ""


@dataclass(frozen=True)
class BinaryScoring(ScoringMode):
    """Binary scoring with configurable true/false values and aggregation."""

    mode: str = "binary"
    true_value: str = "yes"
    false_value: str = "no"
    agg: str = "rate"


@dataclass(frozen=True)
class EnumScoring(ScoringMode):
    """Categorical scoring over a fixed set of values."""

    mode: str = "enum"
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class NumericScoring(ScoringMode):
    """Numeric scoring within a bounded range."""

    mode: str = "numeric"
    min_val: float = 0.0
    max_val: float = 1.0


# ---------------------------------------------------------------------------
# Normalizer factories (applied at extraction boundary)
# ---------------------------------------------------------------------------


def binary_normalizer(true_value: str = "yes", false_value: str = "no") -> Callable[[Any], Any]:
    """Normalize extracted value via classify_binary; passthrough on invalid."""

    def _normalize(value: Any) -> Any:
        c = classify_binary(value, true_value, false_value)
        return c.canonical if c.is_valid else value

    return _normalize


_MCQ_LETTERS = frozenset("ABCDE")


def enum_normalizer(values: tuple[str, ...]) -> Callable[[Any], Any]:
    """Normalize extracted value via classify_enum; passthrough on invalid.

    For MCQ-style enums (single letters A-E), falls back to normalize_mcq
    when exact match fails -- handles 'C) 10', 'the answer is B', etc.
    """
    is_mcq = all(v in _MCQ_LETTERS for v in values)

    def _normalize(value: Any) -> Any:
        c = classify_enum(value, values)
        if c.is_valid:
            return c.canonical
        if is_mcq:
            from .mcq import normalize_mcq

            extracted = normalize_mcq(str(value))
            if extracted is not None:
                c2 = classify_enum(extracted, values)
                if c2.is_valid:
                    return c2.canonical
        return value

    return _normalize


def numeric_normalizer(min_val: float = 0.0, max_val: float = 1.0) -> Callable[[Any], Any]:
    """Normalize extracted value via classify_numeric; passthrough on invalid."""

    def _normalize(value: Any) -> Any:
        c = classify_numeric(value, min_val, max_val)
        return c.value if c.is_valid else value

    return _normalize


def normalizer_for_scoring(scoring: ScoringMode) -> Callable[[Any], Any] | None:
    """Return the appropriate normalizer for a ScoringMode, or None."""
    if isinstance(scoring, BinaryScoring):
        return binary_normalizer(scoring.true_value, scoring.false_value)
    if isinstance(scoring, EnumScoring):
        return enum_normalizer(scoring.values)
    if isinstance(scoring, NumericScoring):
        return numeric_normalizer(scoring.min_val, scoring.max_val)
    return None


# ---------------------------------------------------------------------------
# Scoring dispatch tables
# ---------------------------------------------------------------------------

_SCORING_CLASSES: dict[str, type[ScoringMode]] = {
    "binary": BinaryScoring,
    "enum": EnumScoring,
    "numeric": NumericScoring,
}

_COMPATIBLE_TYPES: dict[str, frozenset[type]] = {
    "binary": frozenset({str}),
    "enum": frozenset({str}),
    "numeric": frozenset({float, int}),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_scoring(raw: Any) -> ScoringMode | None:
    """Parse YAML scoring values into ScoringMode instances.

    - None -> None
    - "binary" -> BinaryScoring()
    - {"mode": "binary", "true_value": "oui"} -> BinaryScoring(true_value="oui")
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw == "enum":
            raise ValueError(
                "Bare 'enum' scoring requires values. "
                "Use {'mode': 'enum', 'values': [...]} instead."
            )
        cls = _SCORING_CLASSES.get(raw)
        if cls is None:
            raise ValueError(f"Unknown scoring mode: {raw!r}")
        return cls()
    if isinstance(raw, dict):
        mode = raw.get("mode")
        if mode is None:
            raise ValueError("Scoring dict must include 'mode' key")
        cls = _SCORING_CLASSES.get(mode)
        if cls is None:
            raise ValueError(f"Unknown scoring mode: {mode!r}")
        kwargs = {k: v for k, v in raw.items() if k != "mode"}
        # Convert list values to tuple for frozen dataclass compatibility
        if "values" in kwargs:
            if isinstance(kwargs["values"], str):
                raise ValueError(
                    f"Scoring 'values' must be a list, "
                    f"got string {kwargs['values']!r}. "
                    f"Use ['a', 'b', 'c'] not 'abc'."
                )
            if isinstance(kwargs["values"], list):
                kwargs["values"] = tuple(kwargs["values"])
        _validate_scoring_kwargs(mode, kwargs)
        return cls(**kwargs)
    raise ValueError(f"Invalid scoring value: {raw!r}")


def _validate_scoring_kwargs(mode: str, kwargs: dict[str, Any]) -> None:
    """Validate scoring kwargs before constructing ScoringMode."""
    if mode == "binary":
        tv = kwargs.get("true_value", "yes")
        fv = kwargs.get("false_value", "no")
        # Non-empty
        if not tv.strip():
            raise ValueError("BinaryScoring true_value must be non-empty")
        if not fv.strip():
            raise ValueError("BinaryScoring false_value must be non-empty")
        # Single token after canon
        tv_canon = _canon_binary(tv)
        fv_canon = _canon_binary(fv)
        if " " in tv_canon:
            raise ValueError(
                f"BinaryScoring true_value must be a single token, got {tv!r}. "
                f"Use EnumScoring for multi-word values."
            )
        if " " in fv_canon:
            raise ValueError(
                f"BinaryScoring false_value must be a single token, got {fv!r}. "
                f"Use EnumScoring for multi-word values."
            )
        # No collision
        if tv_canon == fv_canon:
            raise ValueError(
                f"BinaryScoring true_value and false_value collide after "
                f"canonicalization: {tv!r} and {fv!r} both become {tv_canon!r}"
            )
        # Valid agg
        agg = kwargs.get("agg", "rate")
        if agg not in ("rate", "any"):
            raise ValueError(f"BinaryScoring agg must be 'rate' or 'any', got {agg!r}")

    elif mode == "enum":
        values = kwargs.get("values", ())
        if not values:
            raise ValueError("EnumScoring values must be non-empty")
        if not all(isinstance(v, str) for v in values):
            raise ValueError("EnumScoring values must all be strings")
        # Check canon collisions
        seen: dict[str, str] = {}
        for v in values:
            c = _canon_enum(v)
            if c in seen:
                raise ValueError(
                    f"EnumScoring values collide after canonicalization: "
                    f"{seen[c]!r} and {v!r} both become {c!r}"
                )
            seen[c] = v

    elif mode == "numeric":
        lo = kwargs.get("min_val", 0.0)
        hi = kwargs.get("max_val", 1.0)
        if not math.isfinite(lo):
            raise ValueError(f"NumericScoring min_val must be finite, got {lo}")
        if not math.isfinite(hi):
            raise ValueError(f"NumericScoring max_val must be finite, got {hi}")
        if lo > hi:
            raise ValueError(f"NumericScoring min_val ({lo}) must be <= max_val ({hi})")


def validate_type_scoring(name: str, field_type: type, scoring: ScoringMode) -> None:
    """Check that field type is compatible with its scoring mode. Raises ValueError."""
    compatible = _COMPATIBLE_TYPES.get(scoring.mode)
    if compatible is None:
        raise ValueError(f"Unknown scoring mode: {scoring.mode!r}")
    if field_type not in compatible:
        names = sorted(t.__name__ for t in compatible)
        raise ValueError(
            f"Field {name!r}: type {field_type.__name__} is "
            f"incompatible with scoring mode {scoring.mode!r} "
            f"(requires one of {', '.join(names)})"
        )


# ---------------------------------------------------------------------------
# Field resolution (from resolve.py)
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "bool": bool,
}


class _FieldDesc(TypedDict, total=False):
    type: Required[str]
    description: str
    scoring: str | dict[str, Any]


def _resolve_fields(raw: dict[str, str | _FieldDesc]) -> dict[str, FieldSpec]:
    """Resolve YAML field declarations to FieldSpec objects."""
    result: dict[str, FieldSpec] = {}
    for name, spec in raw.items():
        if isinstance(spec, str):
            if spec not in _TYPE_MAP:
                raise ValueError(
                    f"Unknown field type {spec!r} for {name!r}. Valid: {set(_TYPE_MAP)}"
                )
            result[name] = FieldSpec(_TYPE_MAP[spec])
        else:
            type_str = spec["type"]
            if type_str not in _TYPE_MAP:
                raise ValueError(
                    f"Unknown field type {type_str!r} for {name!r}. Valid: {set(_TYPE_MAP)}"
                )
            ft = _TYPE_MAP[type_str]
            desc = spec.get("description", "")
            scoring = resolve_scoring(spec.get("scoring"))
            if scoring is not None:
                validate_type_scoring(name, ft, scoring)
            normalizer = normalizer_for_scoring(scoring) if scoring is not None else None
            result[name] = FieldSpec(ft, desc, scoring, normalizer)
    return result
