"""Pure helpers for the Lean taskset: starter-file construction, theorem-signature
canonicalization, the reward-hacking signature guard, and compile-output parsing.

Everything here is a pure function over plain strings — no runtime, no sandbox, no
verifiers types — so it's trivially unit-testable and shared between ``setup``
(building the starter file), the reward (the signature guard + compile parsing),
and ``validate`` (planting the gold proof).
"""

from __future__ import annotations

import re

PROTECTED_HEADER_COMMENT = (
    "-- DO NOT MODIFY the theorem statement below. The grader checks\n"
    "-- that the original `theorem ... := by` text still appears in\n"
    "-- this file. Only edit the proof body (currently `sorry`) and\n"
    "-- lines after it."
)


# ── Imports / preamble normalization ─────────────────────────────────────────


def normalize_imports(text: str) -> str:
    """Collapse every ``import Mathlib*`` line to a single ``import Mathlib``.

    Some datasets (miniF2F) ship fine-grained Mathlib imports that don't resolve
    against the monolithic ``import Mathlib`` the sandbox image provides.
    """
    lines = text.split("\n")
    out: list[str] = []
    inserted = False
    for line in lines:
        if line.strip().startswith("import Mathlib"):
            if not inserted:
                out.append("import Mathlib")
                inserted = True
        else:
            out.append(line)
    return "\n".join(out)


def _build_preamble(imports_str: str, header: str, normalize: bool) -> str:
    if header and header.strip().startswith("import"):
        preamble = header.strip()
        return normalize_imports(preamble) if normalize else preamble
    parts = [imports_str.strip()]
    if header and header.strip():
        parts.append(header.strip())
    preamble = "\n\n".join(parts)
    return normalize_imports(preamble) if normalize else preamble


# ── Theorem signature canonicalization ───────────────────────────────────────


def _normalize_signature(stmt: str) -> str:
    """Canonicalize a Lean theorem statement to end with ``:= by``.

    Strips trailing ``sorry``/``admit`` placeholders and any trailing ``by`` /
    ``:=`` tokens, then re-appends `` := by``. Places the appended token on a new
    indented line when the last line of the stripped signature already contains a
    ``--`` comment (otherwise the ``:= by`` would land inside the line comment and
    Lean would silently ignore it).
    """
    s = stmt.rstrip()
    s = re.sub(r"\s*\b(?:sorry|admit)\b\s*$", "", s)
    s = re.sub(r"\s*\bby\b\s*$", "", s)
    s = re.sub(r"\s*:=\s*$", "", s)
    s = s.rstrip()
    last_newline = s.rfind("\n")
    last_line = s[last_newline + 1 :] if last_newline != -1 else s
    sep = "\n    " if "--" in last_line else " "
    return s + sep + ":= by"


def _split_imports_and_signature(stmt: str) -> tuple[str, str]:
    decl_match = re.search(r"^(?:theorem|lemma|example)\s", stmt, flags=re.MULTILINE)
    if not decl_match:
        return "", stmt
    return stmt[: decl_match.start()].rstrip(), stmt[decl_match.start() :]


def expected_protected_signature(formal_statement: str) -> str:
    """Return the canonical ``theorem ... := by`` block the reward pins as ground truth.

    Pure function over ``formal_statement``; the imports/header portion (if the
    statement carries one inline) is stripped first.
    """
    stmt = formal_statement or ""
    if stmt.strip().startswith("import "):
        _, signature_raw = _split_imports_and_signature(stmt)
    else:
        signature_raw = stmt
    return _normalize_signature(signature_raw).strip()


# ── Comment / string stripping for the signature guard ───────────────────────


def strip_lean_comments(text: str) -> str:
    """Remove Lean line/block comments **and string literals** from ``text``.

    Lean comments come in two forms: ``-- ...`` line comments (to end of line) and
    ``/- ... -/`` block comments (nestable; ``/-- ... -/`` doc comments are a
    special case). String literals must also be stripped: a Lean
    ``"theorem ... := by"`` constant or doc-string would otherwise let a model hide
    the pinned signature inside a string while rewriting the live declaration to a
    trivial one, defeating the substring guard. We handle both regular
    double-quoted strings (with backslash escapes) and triple-quoted raw strings.

    Block comments nest, so we count depth; line comments end at the next newline.
    Outside comments and strings, newlines are preserved so the result keeps
    roughly the right shape for substring matching downstream.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    block_depth = 0
    in_line_comment = False
    while i < n:
        ch = text[i]
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append(ch)
            i += 1
            continue
        if block_depth > 0:
            if i + 1 < n and text[i : i + 2] == "-/":
                block_depth -= 1
                i += 2
                continue
            if i + 1 < n and text[i : i + 2] == "/-":
                block_depth += 1
                i += 2
                continue
            if ch == "\n":
                out.append(ch)
            i += 1
            continue
        if i + 1 < n and text[i : i + 2] == "/-":
            block_depth = 1
            i += 2
            continue
        if i + 1 < n and text[i : i + 2] == "--":
            in_line_comment = True
            i += 2
            continue
        # Triple-quoted raw string ``"""..."""`` — skip until the closing
        # triple-quote, preserving newlines.
        if i + 2 < n and text[i : i + 3] == '"""':
            i += 3
            while i < n:
                if i + 2 < n and text[i : i + 3] == '"""':
                    i += 3
                    break
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            continue
        # Regular string ``"..."`` — handle ``\\`` escapes, stop at the next
        # unescaped quote or newline (Lean strings are single-line).
        if ch == '"':
            i += 1
            while i < n and text[i] != '"' and text[i] != "\n":
                if text[i] == "\\" and i + 1 < n:
                    i += 2
                else:
                    i += 1
            if i < n and text[i] == '"':
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def protected_signature_substring_present(
    content: str, expected_signature: str
) -> bool:
    """True when the locked signature text still appears in the file.

    Strips Lean comments from BOTH sides — without that a model could paste the
    pinned signature into a ``--`` or ``/- ... -/`` block while rewriting the live
    declaration to something trivial, and the asymmetric variant would also misfire
    if ``expected_signature`` itself contains a comment. Tries an exact substring
    match first, then a whitespace-flexible match (each side collapsed to
    single-spaced tokens) so the model can re-indent or reflow whitespace freely.
    """
    if not expected_signature:
        return True
    decommented_content = strip_lean_comments(content)
    decommented_expected = strip_lean_comments(expected_signature)
    if not decommented_expected.strip():
        return True
    if decommented_expected in decommented_content:
        return True
    flat_signature = " ".join(decommented_expected.split())
    flat_content = " ".join(decommented_content.split())
    return flat_signature in flat_content


# ── Starter-file construction ────────────────────────────────────────────────


def build_starter_file(
    formal_statement: str,
    header: str = "",
    imports: str = "import Mathlib",
    normalize: bool = False,
    proof_body: str | None = None,
) -> str:
    """Construct the starter proof file.

    Layout: preamble (imports / header) + a brief ``-- DO NOT MODIFY`` comment
    block + the normalized theorem signature + the proof body. If ``proof_body`` is
    None (the default) the body is the placeholder ``  sorry`` — what's planted at
    rollout start. A supplied gold ``proof_body`` (e.g. by ``validate``) replaces
    the placeholder so the file is the full reference solution.
    """
    stmt = formal_statement or ""
    if stmt.strip().startswith("import "):
        imports_block, signature_raw = _split_imports_and_signature(stmt)
        preamble = normalize_imports(imports_block) if normalize else imports_block
    else:
        preamble = _build_preamble(imports or "import Mathlib", header, normalize)
        signature_raw = stmt

    signature = _normalize_signature(signature_raw)
    body = "  sorry" if proof_body is None else proof_body.rstrip()
    wrapped = f"{PROTECTED_HEADER_COMMENT}\n{signature}\n{body}\n"
    if preamble:
        return preamble.rstrip() + "\n\n" + wrapped
    return wrapped


# ── Compile-output parsing ───────────────────────────────────────────────────


def parse_compile_output(output: str) -> tuple[bool, str, int]:
    """Parse a ``lake env lean ...; echo EXIT_CODE:$?`` transcript.

    Returns ``(compiled, cleaned_output, exit_code)`` where ``compiled`` is True iff
    the compiler exited 0 with no ``declaration uses 'sorry'`` diagnostic.

    Matches the LAST ``EXIT_CODE:N`` — that's the one our shell appends at the end
    of the command. Matching the first occurrence would let a model inject
    ``#eval IO.println "EXIT_CODE:0"`` into the proof file to bypass the
    sorry/exit-code checks: the regex would hit the injected marker, truncate
    everything after it (hiding the real ``declaration uses 'sorry'`` diagnostic and
    the real EXIT_CODE), and report success.
    """
    exit_code = 1
    matches = list(re.finditer(r"EXIT_CODE:(\d+)", output))
    if matches:
        last = matches[-1]
        exit_code = int(last.group(1))
        output = output[: last.start()].strip()
    has_sorry = bool(re.search(r"declaration uses 'sorry'", output))
    return (exit_code == 0 and not has_sorry), output, exit_code
