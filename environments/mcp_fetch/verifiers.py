from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import yaml


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def expect_equals(actual: str, expected: str) -> bool:
    return normalize_ws(actual) == normalize_ws(expected)


def expect_contains(text: str, needle: str) -> bool:
    return needle in text


def expect_status(payload: Dict[str, Any], status: int) -> bool:
    return int(payload.get("status", -1)) == status


def expect_header(payload: Dict[str, Any], key: str, expected: str) -> bool:
    headers = {k.lower(): v for k, v in (payload.get("headers") or {}).items()}
    return headers.get(key.lower()) == expected


def expect_json_key(payload: Dict[str, Any], dotted: str, expected: Any) -> bool:
    obj = payload.get("body_json") or {}
    cur: Any = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return False
            if idx < 0 or idx >= len(cur):
                return False
            cur = cur[idx]
        else:
            return False
    return cur == expected


def expect_hash(payload: Dict[str, Any], expected_hex: str) -> bool:
    return str(payload.get("hash")) == expected_hex


def expect_field(payload: Dict[str, Any], field: str, expected: Any) -> bool:
    return payload.get(field) == expected


def expect_bool(payload: Dict[str, Any], field: str, expected: bool) -> bool:
    return bool(payload.get(field)) is expected


def expect_number(payload: Dict[str, Any], field: str, expected: int | float) -> bool:
    value = payload.get(field)
    try:
        return float(value) == float(expected)
    except (TypeError, ValueError):
        return False


def expect_hash_digit_sum(payload: Dict[str, Any], expected: int) -> bool:
    digest = str(payload.get("hash") or "")
    digit_sum = sum(int(ch) for ch in digest if ch.isdigit())
    return digit_sum == int(expected)


def expect_char_count(
    payload: Dict[str, Any],
    field: str,
    *,
    char: str,
    case_insensitive: bool,
    expected: int,
) -> bool:
    target = (char or "")[:1]
    if not target:
        return False
    text = str(_get_field(payload, field) or "")
    haystack = text
    needle = target
    if case_insensitive:
        haystack = haystack.lower()
        needle = needle.lower()
    return haystack.count(needle) == int(expected)


def _get_field(payload: Dict[str, Any], field: str) -> Any:
    if field == "body_text":
        return payload.get("body_text") or ""
    if field == "final_url_suffix":
        final_url = payload.get("final_url") or ""
        return urlparse(final_url).path
    return payload.get(field)


def run_verifier(verifier: Dict[str, Any], payload: Dict[str, Any]) -> Optional[bool]:
    vtype = verifier.get("type")

    if vtype == "equals":
        field = verifier.get("field")
        if not field:
            return False
        actual = _get_field(payload, field)
        return expect_equals(str(actual), str(verifier.get("pattern", "")))
    if vtype == "contains":
        field = verifier.get("field")
        if not field:
            return False
        actual = str(_get_field(payload, field) or "")
        return expect_contains(actual, str(verifier.get("pattern", "")))
    if vtype == "header":
        return expect_header(payload, verifier["key"], verifier["equals"])
    if vtype == "status":
        return expect_status(payload, int(verifier["equals"]))
    if vtype == "json_key":
        return expect_json_key(payload, verifier["path"], verifier["equals"])
    if vtype == "hash":
        return expect_hash(payload, verifier["equals"])
    if vtype == "hash_digit_sum":
        return expect_hash_digit_sum(payload, verifier["equals"])
    if vtype == "char_count":
        return expect_char_count(
            payload,
            verifier["field"],
            char=str(verifier.get("char", "")),
            case_insensitive=bool(verifier.get("case_insensitive", False)),
            expected=verifier["equals"],
        )
    if vtype == "field_bool":
        return expect_bool(payload, verifier["field"], bool(verifier["equals"]))
    if vtype == "field_number":
        return expect_number(payload, verifier["field"], verifier["equals"])
    if vtype == "judge":
        # Defer to JudgeRubric evaluation in the calling context.
        return None

    raise ValueError(f"Unsupported verifier type: {vtype}")


_JUDGE_CACHE: Dict[str, Dict[str, Any]] | None = None


def load_judge_rubrics(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load judge rubric definitions from YAML."""

    global _JUDGE_CACHE
    if _JUDGE_CACHE is not None:
        return _JUDGE_CACHE

    file_path = path or Path(__file__).resolve().parent / "tasks" / "judge_rubrics.yaml"
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    rubrics = data.get("rubrics") or {}
    if not isinstance(rubrics, dict):
        raise ValueError("Invalid judge rubric file: expected top-level 'rubrics' mapping")
    _JUDGE_CACHE = {str(key): value for key, value in rubrics.items()}
    return _JUDGE_CACHE


def get_judge_prompt(rubric_id: str) -> str:
    rubrics = load_judge_rubrics()
    if rubric_id not in rubrics:
        raise KeyError(f"Unknown judge rubric '{rubric_id}'")
    rubric = rubrics[rubric_id]
    prompt = rubric.get("judge_prompt")
    if not prompt:
        raise ValueError(f"Judge rubric '{rubric_id}' missing 'judge_prompt'")
    return str(prompt)
