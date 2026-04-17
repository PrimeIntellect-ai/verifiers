"""MCQ answer normalization.

Callers pass already-parsed (post-``parse_channels``) text. No think-tag
handling lives here.
"""

from __future__ import annotations

import re

_XML_STRIP_RE = re.compile(r"</?[^>]+>")


# Pre-filter patterns (disqualify ambiguous/hedged responses)
_HEDGE_RE = re.compile(r"(?i)\b(not sure|uncertain|cannot determine|i(?:'m| am) unsure)\b")
_MULTI_RE = re.compile(r"(?i)\b(?:both|either|all of)\s+(?:the above|[A-E]\s+(?:and|or)\s+[A-E])\b")
_HEDGE_RANK_RE = re.compile(
    r"(?i)\b(?:most likely|best answer is)\b.*\b(?:but|however|although)\b",
    re.DOTALL,
)

_PRE_FILTERS = [_HEDGE_RE, _MULTI_RE, _HEDGE_RANK_RE]

# Cascade extractors (tried in order; first match wins)
_BARE_MCQ_RE = re.compile(r"^([A-Ea-e])$")
_OPTION_PREFIX_RE = re.compile(r"^\s*\(?([A-Ea-e])\)?\s*[).:\-]")
_ANSWER_FRAME_RE = re.compile(
    r"(?i)(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)"
    r"\s*(?:definitely|probably|clearly|obviously|likely|certainly)?"
    r"\s*\(?([A-Ea-e])\)?"
)
_CHOOSE_FRAME_RE = re.compile(r"(?i)I\s+(?:choose|pick|select|go with)\s+\(?([A-Ea-e])\)?")
_OPTION_LABEL_RE = re.compile(r"(?i)\b(?:option|choice)\s+([A-Ea-e])\b")
_LETTER_RE = re.compile(r"\b([A-E])\b")


def _is_article_a(letter: str, context: str) -> bool:
    if letter != "A":
        return False
    m = re.search(r"\bA\s+(?:lot|few|great|large|small|number|very)\b", context)
    return m is not None


def _extract_bare(cleaned: str) -> str | None:
    m = _BARE_MCQ_RE.match(cleaned)
    return m.group(1).upper() if m else None

def _extract_option_prefix(cleaned: str) -> str | None:
    m = _OPTION_PREFIX_RE.match(cleaned)
    return m.group(1).upper() if m else None

def _extract_answer_frame(cleaned: str) -> str | None:
    m = _ANSWER_FRAME_RE.search(cleaned)
    return m.group(1).upper() if m else None

def _extract_choose_frame(cleaned: str) -> str | None:
    m = _CHOOSE_FRAME_RE.search(cleaned)
    return m.group(1).upper() if m else None

def _extract_option_label(cleaned: str) -> str | None:
    m = _OPTION_LABEL_RE.search(cleaned)
    return m.group(1).upper() if m else None

def _extract_first_line(cleaned: str) -> str | None:
    first_line = cleaned.split("\n")[0].strip()
    if not first_line:
        return None
    matches = _LETTER_RE.findall(first_line)
    candidates = [c for c in matches if not _is_article_a(c, first_line)]
    if len(candidates) == 1:
        return str(candidates[0]).upper()
    return None

def _extract_terminal(cleaned: str) -> str | None:
    all_matches = _LETTER_RE.findall(cleaned)
    candidates = [c for c in all_matches if not _is_article_a(c, cleaned)]
    if candidates:
        return str(candidates[-1]).upper()
    return None


_EXTRACTORS = [
    _extract_bare, _extract_option_prefix, _extract_answer_frame,
    _extract_choose_frame, _extract_option_label, _extract_first_line,
    _extract_terminal,
]


def normalize_mcq(text: str) -> str | None:
    cleaned = _XML_STRIP_RE.sub("", text).strip()
    if not cleaned:
        return None
    for pattern in _PRE_FILTERS:
        if pattern.search(cleaned):
            return None
    for extractor in _EXTRACTORS:
        result = extractor(cleaned)
        if result is not None:
            return result
    return None
