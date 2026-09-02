"""Keep known secret values out of uploads.

Redaction is exact-match only: a known value is replaced with `[REDACTED]` wherever it
appears inside a JSON string, and nothing is guessed from the shape of the text, so
ordinary content is never rewritten. The prime CLI carries the same redactor
(`prime_cli/utils/redact.py`) and owns user-supplied `--secret` values.
"""

import json
import re
from collections.abc import Iterator, Mapping
from urllib.parse import unquote, urlsplit

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 8
SECRET_NAME = re.compile(
    r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|COOKIE|AUTHORIZATION|(?:^|[_-])AUTH(?:[_-]|$)",
    re.IGNORECASE,
)
"""Variable and header names whose values are credentials."""
JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def url_credentials(value: str) -> Iterator[str]:
    """The password — or the bare user token — inside a `scheme://user:password@host`
    value, as written and percent-decoded the way a client uses it. A username next to a
    password is a name, not a secret (`postgres`, the egress proxy's `verifiers`), so a
    token placed there beside a dummy password (GitHub's legacy `token:x-oauth-basic`)
    is not recognised."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return
    if "@" not in parts.netloc:
        return
    # With a password slot, even an empty one, the username is a name, not a token.
    userinfo = parts.username if parts.password is None else parts.password
    if userinfo:
        yield from {userinfo, unquote(userinfo)}


def env_credentials(mapping: Mapping[str, object]) -> Iterator[str]:
    """The credentials in an environment-like mapping (variables, headers): every value
    under a credential-like name, and the URL credentials in any value whatever its name
    (`DATABASE_URL`, `HTTP_PROXY`)."""
    for name, value in mapping.items():
        if not isinstance(value, str):
            continue
        if SECRET_NAME.search(name):
            yield value
        yield from url_credentials(value)


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
