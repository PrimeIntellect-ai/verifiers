"""Credential reduction for in-memory Platform upload payloads."""

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "[REDACTED]"

_PLACEHOLDER = re.compile(
    r"^(?:\[?redacted\]?|masked|dummy|example|test|none|null|changeme|"
    r"replace[_ -]?me|x{4,}|\*{4,}|<[^>]+>|\$\{?[A-Z][A-Z0-9_]*\}?)$",
    re.IGNORECASE,
)
_SECRET_ENV = re.compile(
    r"API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION|COOKIE|"
    r"PRIVATE_?KEY|CONNECTION_STRING|DATABASE_URL|REDIS_URL",
    re.IGNORECASE,
)
_REFERENCE_SUFFIXES = (
    "_env",
    "_env_var",
    "_file",
    "_name",
    "_path",
    "_var",
    "_variable",
)
_SENSITIVE_FIELDS = {
    "api_key",
    "access_token",
    "auth_token",
    "authorization",
    "client_secret",
    "connection_string",
    "cookie",
    "credential",
    "database_url",
    "password",
    "passwd",
    "private_key",
    "proxy_authorization",
    "redis_url",
    "refresh_token",
    "sas_token",
    "secret",
    "secret_key",
    "session_token",
    "signature",
    "token",
}
_SCHEMA_VALUES = {"const", "default", "example", "examples"}
_SCHEMA_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
_AUTH_VALUE = re.compile(r"^(?:bearer|basic)\s+(.+)$", re.IGNORECASE)
_PATTERNS = (
    re.compile(
        r"(?P<secret>-----BEGIN (?P<label>(?:(?:RSA|EC|OPENSSH|DSA|ENCRYPTED) )?"
        r"PRIVATE KEY)-----.*?-----END (?P=label)-----)",
        re.DOTALL,
    ),
    re.compile(
        r"(?<![A-Za-z0-9])(?P<secret>"
        r"sk-(?:ant-|or-v1-)?[A-Za-z0-9_-]{20,}|"
        r"AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"hf_[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}|glpat-[A-Za-z0-9_-]{20,}|"
        r"xox[baprs]-[A-Za-z0-9-]{10,}|npm_[A-Za-z0-9]{20,}|"
        r"pypi-[A-Za-z0-9_-]{30,}|SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,})"
    ),
    re.compile(
        r"(?P<secret>https://(?:hooks\.slack\.com/services|"
        r"discord(?:app)?\.com/api/webhooks)/[^\s\"']+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\b(?:authorization|proxy-authorization|x-api-key|api[_ -]?key|"
        r"access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|client[_ -]?secret|"
        r"secret|password|passwd|cookie|private[_ -]?key|signature)\b\s*[:=]\s*|"
        r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_?KEY|ACCESS_?TOKEN|REFRESH_?TOKEN|"
        r"AUTH_?TOKEN|SESSION_?TOKEN|TOKEN|CLIENT_?SECRET|SECRET(?:_ACCESS_?KEY)?|"
        r"PASSWORD|PASSWD|CREDENTIAL|PRIVATE_?KEY)\s*=\s*|"
        r"--(?:api-key|access-token|auth-token|client-secret|password|private-key|"
        r"secret|token)(?:=|\s+))(?:(?:bearer|basic)\s+)?[\"']?"
        r"(?P<secret>[^\s,;\"']{8,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"[A-Za-z][A-Za-z0-9+.-]*://(?:[^:/@\s]+:)?(?P<secret>[^/@\s]{8,})@|"
        r"[?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|sig|"
        r"signature|credential|authorization|auth|password)="
        r"(?P<secret_query>[^&#\s\"']{8,})",
        re.IGNORECASE,
    ),
)


def _normalize(name: str) -> str:
    return (
        re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
                "_",
                name,
            ),
        )
        .strip("_")
        .lower()
    )


def _sensitive(name: str) -> bool:
    name = _normalize(name)
    return not name.endswith(_REFERENCE_SUFFIXES) and any(
        name == field or name.endswith(f"_{field}") for field in _SENSITIVE_FIELDS
    )


def _is_secret(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= 8
        and not _PLACEHOLDER.fullmatch(value.strip())
    )


def prepare_upload(value: Any, known_secrets: Iterable[str] = ()) -> tuple[Any, int]:
    """Return a reduced copy and the number of credential-bearing values changed."""
    secrets = {
        secret
        for secret in (
            *known_secrets,
            *(
                value
                for name, value in os.environ.items()
                if _SECRET_ENV.search(name)
                and not _normalize(name).endswith(_REFERENCE_SUFFIXES)
            ),
        )
        if _is_secret(secret)
    }

    def remember(candidate: Any) -> None:
        if _is_secret(candidate):
            secrets.add(candidate)
            if (match := _AUTH_VALUE.fullmatch(candidate.strip())) and _is_secret(
                token := match.group(1)
            ):
                secrets.add(token)
        elif isinstance(candidate, Mapping):
            for item in candidate.values():
                remember(item)
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                remember(item)

    def discover(
        child: Any, schema_secret: bool = False, properties: bool = False
    ) -> None:
        if isinstance(child, Mapping):
            for key, nested in child.items():
                name = _normalize(str(key))
                if properties:
                    schema = isinstance(nested, Mapping) and (
                        nested.get("type") in _SCHEMA_TYPES
                        or any(
                            field in nested
                            for field in ("$ref", "allOf", "anyOf", "oneOf")
                        )
                    )
                    if schema:
                        discover(nested, _sensitive(name))
                    else:
                        if _sensitive(name):
                            remember(nested)
                        discover(nested)
                    continue
                if _sensitive(name) or schema_secret and name in _SCHEMA_VALUES:
                    remember(nested)
                discover(nested, schema_secret, name == "properties")
        elif isinstance(child, (list, tuple)):
            for nested in child:
                discover(nested, schema_secret)

    discover(value)
    replacements = 0

    def redact_text(text: str) -> str:
        nonlocal replacements
        for secret in sorted(secrets, key=len, reverse=True):
            if secret in text:
                replacements += 1
                text = text.replace(secret, REDACTED)
        for pattern in _PATTERNS:

            def replace(match: re.Match[str]) -> str:
                nonlocal replacements
                group = (
                    "secret"
                    if match.groupdict().get("secret") is not None
                    else "secret_query"
                )
                if not _is_secret(match.group(group)):
                    return match.group(0)
                replacements += 1
                start, end = match.span(group)
                offset = match.start()
                return (
                    f"{match.group(0)[: start - offset]}{REDACTED}"
                    f"{match.group(0)[end - offset :]}"
                )

            text = pattern.sub(replace, text)
        return text

    def reduce(child: Any) -> Any:
        if isinstance(child, Mapping):
            reduced = {}
            for key, nested in child.items():
                safe_key = redact_text(key) if isinstance(key, str) else key
                if safe_key in reduced:
                    raise ValueError(
                        "credential reduction would create duplicate object keys"
                    )
                reduced[safe_key] = reduce(nested)
            return reduced
        if isinstance(child, list):
            return [reduce(nested) for nested in child]
        if isinstance(child, tuple):
            return tuple(reduce(nested) for nested in child)
        return redact_text(child) if isinstance(child, str) else child

    reduced = reduce(value)
    count = replacements
    reduce(reduced)
    if replacements != count:
        raise ValueError("reduced upload still contains credentials")
    return reduced, count
