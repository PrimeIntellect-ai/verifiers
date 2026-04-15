"""Canonical think-tag handling for debate recipe.

Pure leaf module — stdlib only, zero debate imports.
"""

from __future__ import annotations

import functools
import re


@functools.lru_cache(maxsize=16)
def _think_patterns(tag: str) -> tuple[re.Pattern, re.Pattern]:
    """Compile (closed, unclosed) think-tag regexes for a tag name. LRU-cached.

    ``think`` and ``thinking`` are treated as aliases — either tag matches
    both ``<think>`` and ``<thinking>`` so models can use either form.
    Both patterns permit attributes on the opener and closer (``[^>]*``).
    """
    if tag in ("think", "thinking"):
        pattern = r"think(?:ing)?"
    else:
        pattern = re.escape(tag)
    closed = re.compile(
        rf"<{pattern}[^>]*>(.*?)</{pattern}[^>]*>", re.DOTALL | re.IGNORECASE
    )
    unclosed = re.compile(rf"<{pattern}[^>]*>(.*)$", re.DOTALL | re.IGNORECASE)
    return closed, unclosed


# Design: two separate contracts, two functions — no flags.
#
# ``strip_think`` is the PARSING-SAFE path. It removes only *closed* think
# tags; unclosed ``<think...>`` tokens are preserved verbatim. This is
# deliberate (see G6 post-mortem): the previous unclosed-regex approach
# treated prose like ``<think carefully> about B`` as an unclosed opener and
# ate the downstream answer, silently destroying correctness. Parsing callers
# (field extraction) rely on this preservation so that downstream extractors
# can fail loud on malformed output rather than be handed quietly-rewritten
# text.
#
# ``redact_think`` is the PRIVACY-SAFE path. It removes closed tags AND
# aggressively strips from any residual unclosed ``<think...>`` opener to
# end-of-text. Privacy callers (opponent-view history formatting) rely on
# this aggression so an actor emitting ``<think>secret<answer>A</answer>``
# without closing their think tag cannot leak ``secret`` to the opponent's
# view. The author's own unredacted copy is preserved separately on their
# own turn, so nothing legitimate is lost.


def strip_think(text: str, *, tag: str = "thinking") -> tuple[str, str | None]:
    """Strip *closed* think tags. Parsing-safe contract.

    Returns ``(cleaned_text, thinking_content)``. Unclosed openers are left
    in place so downstream parsers can fail loud on malformed output.
    """
    closed_re, _ = _think_patterns(tag)
    matches = closed_re.findall(text)
    cleaned = closed_re.sub("", text).strip()
    if not matches:
        return cleaned, None
    reasoning = "\n".join(part.strip() for part in matches if part.strip())
    return cleaned, reasoning or None


def redact_think(text: str, *, tag: str = "thinking") -> tuple[str, str | None]:
    """Strip think tags with PRIVACY-FIRST semantics.

    Removes closed tags normally and additionally strips from any residual
    unclosed ``<think...>`` opener to end-of-text. Use this when formatting
    an agent's output for another agent that must not see private reasoning.

    Aggression is intentional: even prose like ``<think carefully> about B``
    gets redacted past the opener. In the privacy path that's correct — the
    author already has their own unredacted copy on their own turn, and we
    would rather lose a few trailing prose words than leak real reasoning
    from a model that fails to close a think tag before committing its
    answer.

    Returns ``(cleaned_text, concatenated_thoughts_or_None)``.
    """
    closed_re, unclosed_re = _think_patterns(tag)
    matches = list(closed_re.findall(text))
    cleaned = closed_re.sub("", text)
    unclosed = unclosed_re.search(cleaned)
    if unclosed:
        matches.append(unclosed.group(1))
        cleaned = cleaned[: unclosed.start()]
    reasoning = "\n".join(part.strip() for part in matches if part.strip())
    return cleaned.strip(), reasoning or None
