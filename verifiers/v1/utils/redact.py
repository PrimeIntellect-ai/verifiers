"""Keep known secret values out of uploads.

Redaction is exact-match only: a known value is replaced with `[REDACTED]` wherever it
appears inside a JSON string, and nothing is guessed from the shape of the text, so
ordinary content is never rewritten. The prime CLI carries the same redactor
(`prime_cli/utils/redact.py`) and owns user-supplied `--secret` values.
"""

import json
import re

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 8
SECRET_NAME = re.compile(
    r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE|AUTHORIZATION|(?:^|[_-])AUTH(?:[_-]|$)",
    re.IGNORECASE,
)
"""Variable and header names whose values are credentials."""
JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


class Redactor:
    """Replaces every occurrence of the secrets inside JSON strings, counting hits."""

    def __init__(self, secrets: set[str]) -> None:
        # The string itself is decoded before matching. A JSON document quoted inside it
        # (a tool result) is escaped once per nesting level, by an encoder that may also
        # escape `/`: match those spellings too.
        forms = set(secrets)
        forms |= {form.replace("/", r"\/") for form in forms}
        for _ in range(2):
            forms |= {
                json.dumps(form, ensure_ascii=escape)[1:-1]
                for form in list(forms)
                for escape in (True, False)
            }
        alternatives = "|".join(
            re.escape(form) for form in sorted(forms, key=len, reverse=True)
        )
        self.pattern = re.compile(alternatives) if forms else None
        self.count = 0

    def json(self, text: str) -> str:
        """Redact one JSON document given as text; structure and non-string values stay,
        and so do the bytes of every string without a hit."""
        pattern = self.pattern
        if pattern is None:
            return text

        def string(match: re.Match[str]) -> str:
            token = match.group(0)
            redacted, hits = pattern.subn(REDACTED, json.loads(token))
            if not hits:
                return token
            self.count += hits
            return json.dumps(redacted, ensure_ascii=token.isascii())

        return JSON_STRING.sub(string, text)
