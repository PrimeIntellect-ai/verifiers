"""Credential reduction for in-memory Platform upload payloads."""

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "[REDACTED]"

_PLACEHOLDER = re.compile(
    r"^(?:\[?redacted(?:[_ -]?\d+)?\]?|masked|dummy|example|test|none|null|changeme|"
    r"replace[_ -]?me|x{4,}|\*{4,}|<[^>]+>|\$\{?[A-Z][A-Z0-9_]*\}?)$",
    re.IGNORECASE,
)
_SECRET_ENV = re.compile(
    r"(?:^|_)AUTH(?:$|_)|API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|"
    r"AUTHORIZATION|COOKIE|"
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
    "access_key",
    "access_key_id",
    "account_key",
    "api_key",
    "api_token",
    "access_token",
    "auth_token",
    "auth",
    "authentication",
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
    "secret_access_key",
    "secret_key",
    "session_token",
    "signature",
    "token",
}
_SCHEMA_VALUES = {"const", "default", "enum", "example", "examples"}
_SCHEMA_MARKERS = {
    "$defs",
    "$ref",
    "$schema",
    "allOf",
    "anyOf",
    "items",
    "oneOf",
    "required",
}
_AUTH_VALUE = re.compile(r"^(?:bearer|basic|token)\s+(.+)$", re.IGNORECASE)
_PATTERNS = (
    re.compile(
        r"(?:(?<![A-Za-z0-9_])[\"']?(?i:(?:[A-Za-z][A-Za-z0-9]*[_-]+)*"
        r"(?:x-api-key|api[_ -]?(?:key|token)|account[_ -]?key|"
        r"access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|session[_ -]?token|"
        r"client[_ -]?secret|secret[_ -]?access[_ -]?key|password|passwd|"
        r"credential|private[_ -]?key|signature))\b[\"']?\s*[:=]\s*|"
        r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_?KEY|API_?TOKEN|ACCESS_?TOKEN|"
        r"REFRESH_?TOKEN|AUTH_?TOKEN|SESSION_?TOKEN|TOKEN|CLIENT_?SECRET|"
        r"SECRET(?:_ACCESS_?KEY)?|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_?KEY)\s*=\s*|"
        r"--(?:api-key|api-token|access-token|auth-token|client-secret|password|"
        r"private-key|secret|token)(?:=|\s+))"
        r"[\"']?(?:(?i:bearer|basic|token)\s+)?(?P<secret>[^\s,;\"']{8,})"
    ),
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
        r"xox[baprs]-[A-Za-z0-9-]{10,}|x(?:wfp|app)-[A-Za-z0-9-]{10,}|"
        r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}|npm_[A-Za-z0-9]{20,}|"
        r"pypi-[A-Za-z0-9_-]{30,}|SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,})"
    ),
    re.compile(
        r"(?P<secret>https://(?:hooks\.slack\.com/services|"
        r"discord(?:app)?\.com/api/webhooks)/[^\s\"']+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9_])[\"']?(?:authorization|proxy-authorization)"
        r"[\"']?\s*[:=]\s*[\"']?(?:(?:bearer|basic|token)\s+)?"
        r"(?P<secret>[^\s,;\"']{8,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<![A-Za-z0-9_-])[\"']?cookie[\"']?\s*:\s*[\"']?"
        r"(?P<secret>[^\r\n\"']{8,})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?<![A-Za-z0-9_])[\"']?(?:[A-Za-z][A-Za-z0-9]*[_-]+)*"
        r"(?:x-api-key|api[_ -]?(?:key|token)|account[_ -]?key|"
        r"access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|client[_ -]?secret|"
        r"access[_ -]?key(?:[_ -]?id)?|secret(?:[_ -]?access[_ -]?key)?|"
        r"password|passwd|cookie|private[_ -]?key|signature)\b[\"']?\s*[:=]\s*|"
        r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_?KEY|ACCESS_?TOKEN|REFRESH_?TOKEN|"
        r"AUTH_?TOKEN|SESSION_?TOKEN|TOKEN|CLIENT_?SECRET|SECRET(?:_ACCESS_?KEY)?|"
        r"PASSWORD|PASSWD|CREDENTIAL|PRIVATE_?KEY)\s*=\s*|"
        r"--(?:api-key|access-token|auth-token|client-secret|password|private-key|"
        r"secret|token)(?:=|\s+))"
        r"(?:\"(?P<secret_double>(?:\\.|[^\"\\\r\n]){16,})\"|"
        r"'(?P<secret_single>(?:\\.|[^'\\\r\n]){16,})'|"
        r"(?:(?:bearer|basic|token)\s+)?(?P<secret>[^\s,;\"']{16,}))",
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
    singular = name.removesuffix("s")
    plural_secret = singular != name and any(
        singular == field or singular.endswith(f"_{field}")
        for field in _SENSITIVE_FIELDS - {"token"}
    )
    names = (name, singular) if plural_secret else (name,)
    return not any(
        candidate.endswith(_REFERENCE_SUFFIXES) for candidate in names
    ) and any(
        candidate == field or candidate.endswith(f"_{field}")
        for candidate in names
        for field in _SENSITIVE_FIELDS
    )


def _is_secret(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= 8
        and not _PLACEHOLDER.fullmatch(value.strip())
    )


def prepare_upload(
    value: Any,
    known_secrets: Iterable[str] = (),
    secret_sources: Iterable[Mapping[str, str]] = (),
) -> tuple[Any, int]:
    """Return a reduced copy and the number of credential-bearing values changed."""
    secrets = {
        value
        for name, value in os.environ.items()
        if _SECRET_ENV.search(name)
        and not _normalize(name).endswith(_REFERENCE_SUFFIXES)
        and _is_secret(value)
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
        child: Any,
        schema_secret: bool = False,
        schema_context: bool = False,
        properties: bool = False,
    ) -> None:
        if isinstance(child, Mapping):
            schema_type = child.get("type")
            object_schema = (
                schema_context
                or any(field in child for field in _SCHEMA_MARKERS)
                or "properties" in child
                and (
                    schema_type == "object"
                    or isinstance(schema_type, (list, tuple))
                    and "object" in schema_type
                )
            )
            for key, nested in child.items():
                name = _normalize(str(key))
                if properties:
                    if isinstance(nested, (Mapping, bool)):
                        discover(nested, _sensitive(name), True)
                    else:
                        if _sensitive(name):
                            remember(nested)
                        discover(nested)
                    continue
                if _sensitive(name) or schema_secret and name in _SCHEMA_VALUES:
                    remember(nested)
                discover(
                    nested,
                    schema_secret,
                    object_schema or name in {"schema", "json_schema"},
                    object_schema and name == "properties",
                )
        elif isinstance(child, (list, tuple)):
            for nested in child:
                discover(nested, schema_secret, schema_context, properties)
        elif isinstance(child, str):
            for pattern in _PATTERNS:
                for match in pattern.finditer(child):
                    secret = next(
                        value
                        for name, value in match.groupdict().items()
                        if name.startswith("secret") and value is not None
                    )
                    if _is_secret(secret):
                        secrets.add(secret)

    for secret in known_secrets:
        remember(secret)
    for source in secret_sources:
        for name, secret in source.items():
            if _sensitive(name):
                remember(secret)
    discover(value)
    replacements = 0
    exact = (
        re.compile(
            "|".join(
                re.escape(secret) for secret in sorted(secrets, key=len, reverse=True)
            )
        )
        if secrets
        else None
    )

    def redact_text(text: str) -> str:
        nonlocal replacements
        if exact and (matches := {match.group() for match in exact.finditer(text)}):
            replacements += len(matches)
            text = exact.sub(REDACTED, text)
        for pattern in _PATTERNS:

            def replace(match: re.Match[str]) -> str:
                nonlocal replacements
                group = next(
                    name
                    for name, value in match.groupdict().items()
                    if name.startswith("secret") and value is not None
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

    def reduce(
        child: Any,
        structured_secret: bool = False,
        schema_secret: bool = False,
        schema_context: bool = False,
        properties: bool = False,
    ) -> Any:
        nonlocal replacements
        if isinstance(child, Mapping):
            reduced = {}
            schema_type = child.get("type")
            object_schema = (
                schema_context
                or any(field in child for field in _SCHEMA_MARKERS)
                or "properties" in child
                and (
                    schema_type == "object"
                    or isinstance(schema_type, (list, tuple))
                    and "object" in schema_type
                )
            )
            for key, nested in child.items():
                if structured_secret and _is_secret(key):
                    safe_key = REDACTED
                    replacements += 1
                    suffix = 2
                    while safe_key in reduced:
                        safe_key = f"[REDACTED {suffix}]"
                        suffix += 1
                else:
                    safe_key = redact_text(key) if isinstance(key, str) else key
                if safe_key in reduced:
                    raise ValueError(
                        "credential reduction would create duplicate object keys"
                    )
                normalized = _normalize(str(key))
                property_schema = properties and isinstance(nested, (Mapping, bool))
                nested_secret = structured_secret or (
                    not property_schema
                    and (
                        _sensitive(str(key))
                        or schema_secret
                        and normalized in _SCHEMA_VALUES
                    )
                )
                reduced[safe_key] = reduce(
                    nested,
                    nested_secret,
                    _sensitive(str(key)) if property_schema else schema_secret,
                    property_schema
                    or object_schema
                    or normalized in {"schema", "json_schema"},
                    not structured_secret
                    and object_schema
                    and normalized == "properties",
                )
            return reduced
        if isinstance(child, list):
            return [
                reduce(
                    nested,
                    structured_secret,
                    schema_secret,
                    schema_context,
                    properties,
                )
                for nested in child
            ]
        if isinstance(child, tuple):
            return tuple(
                reduce(
                    nested,
                    structured_secret,
                    schema_secret,
                    schema_context,
                    properties,
                )
                for nested in child
            )
        if (
            structured_secret
            and child is not None
            and not isinstance(child, bool)
            and (
                not isinstance(child, str)
                or child.strip()
                and not _PLACEHOLDER.fullmatch(child.strip())
            )
        ):
            replacements += 1
            return REDACTED
        if not isinstance(child, str):
            return child
        return redact_text(child)

    reduced = reduce(value)
    count = replacements
    reduce(reduced)
    if replacements != count:
        raise ValueError("reduced upload still contains credentials")
    return reduced, count
