"""Keep known secret values out of uploads.

Redaction is exact-match only: a known value is replaced with `[REDACTED]` wherever it
appears inside a JSON string, and nothing is guessed from the shape of the text, so
ordinary content is never rewritten. The prime CLI carries the same redactor
(`prime_cli/utils/redact.py`) and owns user-supplied `--secret` values.
"""

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import unquote, unquote_plus, urlsplit

REDACTED = "[REDACTED]"
MIN_SECRET_LENGTH = 8
SECRET_NAME = re.compile(
    r"(?:(?<![A-Za-z0-9])"
    r"(?:API_?KEYS?|KEYS?|CREDENTIALS?|COOKIES?|AUTHORIZATION|AUTH|SIGNATURES?|SIG|PATS?)"
    r"|TOKENS?|SECRETS?|PASSW(?:OR)?DS?)"
    r"(?:[_-][0-9]+)?$",
    re.IGNORECASE,
)
"""Variable and header names whose values are credentials: the name's last word is a
credential word (`HF_TOKEN`, `X-Api-Key`, `PGPASSWORD`), optionally numbered
(`API_KEY_2`). Compound names are head-final, so `TOKEN_URL`, `COOKIE_DOMAIN`,
`KEY_FILE`, `AUTHORIZATION_URL`, and `SSH_AUTH_SOCK` name metadata about a credential
rather than one, and `KEYCLOAK_REALM` or `TOKENIZERS_PARALLELISM` never named one. A
name whose head word is not a credential word (`SECRET_KEY_BASE`) is missed on
purpose. No other word ends in TOKEN, SECRET, or PASSWORD, so those may also end a
segment (`PGPASSWORD`, `ACCESSTOKEN`)."""
JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def url_credentials(value: str) -> Iterator[str]:
    """The credentials inside a URL, as written and percent-decoded the way a client uses
    them: the password — or the bare user token — of `scheme://user:password@host`, and
    credential-named query values (`?token=…`). A username next to a password is a name,
    not a secret (`postgres`), so a token placed there beside a dummy password (GitHub's
    legacy `token:x-oauth-basic`) is not recognised. Prose is never a URL here: the
    value must start with `scheme://host`."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return
    if not (parts.scheme and parts.netloc):
        return
    if "@" in parts.netloc:
        # With a password slot, even an empty one, the username is a name, not a token.
        userinfo = parts.username if parts.password is None else parts.password
        if userinfo:
            yield from {userinfo, unquote(userinfo)}
    for pair in parts.query.split("&"):
        name, _, raw = pair.partition("=")
        if raw and SECRET_NAME.search(unquote(name)):
            yield from {raw, unquote(raw), unquote_plus(raw)}


def env_credentials(mapping: Mapping[Any, Any]) -> Iterator[str]:
    """The credentials in an environment-like mapping (variables, headers): every value
    under a credential-like name, the URL credentials in any value whatever its name
    (`DATABASE_URL`, `HTTP_PROXY`), and the same again inside a value that is a JSON
    object (`DOCKER_AUTH_CONFIG`, a service-account blob), which is a mapping too."""
    for name, value in mapping.items():
        if isinstance(value, Mapping):
            yield from env_credentials(value)
            continue
        if not isinstance(value, str):
            continue
        if SECRET_NAME.search(name):
            yield value
        yield from url_credentials(value)
        if value.startswith("{"):
            try:
                nested = json.loads(value)
            except ValueError:
                continue
            if isinstance(nested, Mapping):
                yield from env_credentials(nested)


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
