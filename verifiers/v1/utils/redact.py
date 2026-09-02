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
        alternatives = "|".join(
            re.escape(secret) for secret in sorted(secrets, key=len, reverse=True)
        )
        self.pattern = re.compile(alternatives) if secrets else None
        self.count = 0

    def json(self, text: str) -> str:
        """Redact one JSON document given as text; structure and non-string values stay,
        and so do the bytes of every string without a hit. Each string is decoded before
        matching and searched again for quoted JSON inside it (a tool result), so every
        escape spelling at every nesting depth is matched."""
        pattern = self.pattern
        if pattern is None:
            return text

        def string(match: re.Match[str]) -> str:
            token = match.group(0)
            try:
                value = json.loads(token)
            except ValueError:  # quotes in prose, not a JSON string
                return token
            redacted, hits = pattern.subn(REDACTED, self.json(value))
            if redacted == value:
                return token
            self.count += hits
            return json.dumps(redacted, ensure_ascii=token.isascii())

        return JSON_STRING.sub(string, text)
