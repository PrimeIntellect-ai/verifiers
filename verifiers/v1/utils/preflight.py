"""Credential reduction for in-memory Platform upload payloads."""

import os
import re
from collections.abc import Iterable, Mapping
from hashlib import sha256
from typing import Any

REDACTED = "[REDACTED]"
SecretFingerprint = tuple[int, str]

PLACEHOLDER = re.compile(
    r"^(?:\[?redacted(?:[_ -]?\d+)?\]?|masked|dummy|example|test|none|null|changeme|"
    r"replace[_ -]?me|x{4,}|\*{4,}|<[^>]+>|\$\{?[A-Z][A-Z0-9_]*\}?)$",
    re.IGNORECASE,
)
SECRET_ENV = re.compile(
    r"(?:^|_)AUTH(?:$|_)|API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|"
    r"AUTHORIZATION|COOKIE|"
    r"PRIVATE_?KEY|CONNECTION_STRING|DATABASE_URL|REDIS_URL",
    re.IGNORECASE,
)
REFERENCE_SUFFIXES = (
    "_env",
    "_env_var",
    "_file",
    "_name",
    "_path",
    "_var",
    "_variable",
)
SENSITIVE_FIELDS = {
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
SCHEMA_VALUES = {"const", "default", "enum", "example", "examples"}
SCHEMA_MARKERS = {
    "$defs",
    "$ref",
    "$schema",
    "allOf",
    "anyOf",
    "oneOf",
}
AUTH_VALUE = re.compile(r"^(?:bearer|basic|token)\s+(.+)$", re.IGNORECASE)
PATTERNS = (
    re.compile(
        r"(?:(?<![A-Za-z0-9_])[\"']?(?i:(?:[A-Za-z][A-Za-z0-9]*[_-]+)*"
        r"(?:x-api-key|api[_ -]?(?:key|token)|account[_ -]?key|"
        r"access[_ -]?token|refresh[_ -]?token|auth[_ -]?token|session[_ -]?token|"
        r"client[_ -]?secret|secret[_ -]?access[_ -]?key|password|passwd|"
        r"cookie|credential|private[_ -]?key|sas[_ -]?token|signature))\b"
        r"[\"']?\s*[:=]\s*|"
        r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_?KEY|API_?TOKEN|ACCESS_?TOKEN|"
        r"REFRESH_?TOKEN|AUTH_?TOKEN|SESSION_?TOKEN|TOKEN|CLIENT_?SECRET|"
        r"SECRET(?:_ACCESS_?KEY|_?KEY)?|PASSWORD|PASSWD|COOKIE|CREDENTIAL|"
        r"PRIVATE_?KEY|SAS_?TOKEN)\s*=\s*|"
        r"--(?:api-key|api-token|access-token|auth-token|client-secret|cookie|"
        r"password|private-key|sas-token|secret|token)(?:=|\s+))"
        r"(?:\\?\"(?P<secret_short_double>(?:\\.|[^\"\\\r\n]){8,})\\?\"|"
        r"\\?'(?P<secret_short_single>(?:\\.|[^'\\\r\n]){8,})\\?'|"
        r"(?:(?i:bearer|basic|token)\s+)?(?P<secret>[^\s,;\"']{8,}))"
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
        r"(?<![A-Za-z0-9_])\\?[\"']?(?:authorization|proxy-authorization)"
        r"\\?[\"']?\s*[:=]\s*\\?[\"']?(?:(?:bearer|basic|token)\s+)?"
        r"(?P<secret>[^\s,;\\\"']{8,})",
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
        r"access[_ -]?key(?:[_ -]?id)?|secret(?:[_ -]?(?:access[_ -]?)?key)?|"
        r"password|passwd|cookie|private[_ -]?key|sas[_ -]?token|signature)\b"
        r"[\"']?\s*[:=]\s*|"
        r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_?KEY|ACCESS_?TOKEN|REFRESH_?TOKEN|"
        r"AUTH_?TOKEN|SESSION_?TOKEN|TOKEN|CLIENT_?SECRET|SECRET(?:_ACCESS_?KEY)?|"
        r"PASSWORD|PASSWD|COOKIE|CREDENTIAL|PRIVATE_?KEY|SAS_?TOKEN|SECRET_?KEY)"
        r"\s*=\s*|"
        r"--(?:api-key|access-token|auth-token|client-secret|cookie|password|"
        r"private-key|sas-token|secret|token)(?:=|\s+))"
        r"(?:\\?\"(?P<secret_double>(?:\\.|[^\"\\\r\n]){16,})\\?\"|"
        r"\\?'(?P<secret_single>(?:\\.|[^'\\\r\n]){16,})\\?'|"
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


def normalize(name: str) -> str:
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


def is_sensitive(name: str) -> bool:
    name = normalize(name)
    singular = name.removesuffix("s")
    plural_secret = singular != name and any(
        singular == field or singular.endswith(f"_{field}")
        for field in SENSITIVE_FIELDS - {"token"}
    )
    names = (name, singular) if plural_secret else (name,)
    return not any(
        candidate.endswith(REFERENCE_SUFFIXES) for candidate in names
    ) and any(
        f"_{field}_" in f"_{candidate}_"
        or candidate.replace("_", "") == field.replace("_", "")
        for candidate in names
        for field in SENSITIVE_FIELDS
    )


def is_secret(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) >= 8
        and not PLACEHOLDER.fullmatch(value.strip())
    )


def fingerprint_secret(secret: str) -> SecretFingerprint:
    """Non-plaintext material used to find an echoed transport capability after resume."""
    return len(secret), sha256(secret.encode()).hexdigest()


class CredentialReducer:
    def __init__(
        self,
        known_secrets: Iterable[str] = (),
        secret_sources: Iterable[Mapping[str, str]] = (),
        secret_fingerprints: Iterable[SecretFingerprint] = (),
    ) -> None:
        self.secrets = {
            value
            for name, value in os.environ.items()
            if SECRET_ENV.search(name)
            and not normalize(name).endswith(REFERENCE_SUFFIXES)
            and is_secret(value)
        }
        self.fingerprints: dict[int, set[str]] = {}
        self.replacements = 0
        self.exact: re.Pattern[str] | None = None
        for secret in known_secrets:
            self.remember(secret)
        for source in secret_sources:
            for name, secret in source.items():
                if is_sensitive(name):
                    self.remember(secret)
        for length, digest in secret_fingerprints:
            self.fingerprints.setdefault(length, set()).add(digest)

    def remember(self, candidate: Any) -> None:
        if is_secret(candidate):
            self.secrets.add(candidate)
            match = AUTH_VALUE.fullmatch(candidate.strip())
            if match and is_secret(token := match.group(1)):
                self.secrets.add(token)
        elif isinstance(candidate, Mapping):
            for item in candidate.values():
                self.remember(item)
        elif isinstance(candidate, (list, tuple)):
            for item in candidate:
                self.remember(item)

    def discover(
        self,
        value: Any,
        schema_secret: bool = False,
        schema_context: bool = False,
        properties: bool = False,
    ) -> None:
        if isinstance(value, Mapping):
            schema_type = value.get("type")
            object_schema = (
                schema_context
                or any(field in value for field in SCHEMA_MARKERS)
                or "properties" in value
                and (
                    schema_type == "object"
                    or isinstance(schema_type, (list, tuple))
                    and "object" in schema_type
                )
                or schema_type == "array"
                and "items" in value
            )
            for key, child in value.items():
                name = normalize(str(key))
                if properties:
                    if isinstance(child, (Mapping, bool)):
                        self.discover(child, is_sensitive(name), True)
                    else:
                        if is_sensitive(name):
                            self.remember(child)
                        self.discover(child)
                    continue
                if is_sensitive(name) or schema_secret and name in SCHEMA_VALUES:
                    self.remember(child)
                self.discover(
                    child,
                    schema_secret,
                    object_schema or name in {"schema", "json_schema"},
                    object_schema and name == "properties",
                )
        elif isinstance(value, (list, tuple)):
            for child in value:
                self.discover(child, schema_secret, schema_context, properties)
        elif isinstance(value, str):
            for pattern in PATTERNS[:-1]:
                for match in pattern.finditer(value):
                    secret = next(
                        matched
                        for name, matched in match.groupdict().items()
                        if name.startswith("secret") and matched is not None
                    )
                    if is_secret(secret):
                        self.secrets.add(secret)

    def find_fingerprinted(self, text: str) -> set[str]:
        fingerprinted = set()
        for length, digests in self.fingerprints.items():
            for start in range(len(text) - length + 1):
                candidate = text[start : start + length]
                if sha256(candidate.encode()).hexdigest() in digests:
                    fingerprinted.add(candidate)
        return fingerprinted

    def replace_shape(self, match: re.Match[str]) -> str:
        group = next(
            name
            for name, value in match.groupdict().items()
            if name.startswith("secret") and value is not None
        )
        if not is_secret(match.group(group)):
            return match.group(0)
        self.replacements += 1
        start, end = match.span(group)
        offset = match.start()
        return (
            f"{match.group(0)[: start - offset]}{REDACTED}"
            f"{match.group(0)[end - offset :]}"
        )

    def redact_text(self, text: str) -> str:
        if self.exact and (
            matches := {match.group() for match in self.exact.finditer(text)}
        ):
            self.replacements += len(matches)
            text = self.exact.sub(REDACTED, text)
        fingerprinted = self.find_fingerprinted(text)
        self.replacements += len(fingerprinted)
        for secret in sorted(fingerprinted, key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        for pattern in PATTERNS:
            text = pattern.sub(self.replace_shape, text)
        return text

    def reduce(
        self,
        value: Any,
        structured_secret: bool = False,
        schema_secret: bool = False,
        schema_context: bool = False,
        properties: bool = False,
    ) -> Any:
        if isinstance(value, Mapping):
            reduced = {}
            schema_type = value.get("type")
            object_schema = (
                schema_context
                or any(field in value for field in SCHEMA_MARKERS)
                or "properties" in value
                and (
                    schema_type == "object"
                    or isinstance(schema_type, (list, tuple))
                    and "object" in schema_type
                )
                or schema_type == "array"
                and "items" in value
            )
            for key, child in value.items():
                if structured_secret and is_secret(key):
                    safe_key = REDACTED
                    self.replacements += 1
                    suffix = 2
                    while safe_key in reduced:
                        safe_key = f"[REDACTED {suffix}]"
                        suffix += 1
                else:
                    safe_key = self.redact_text(key) if isinstance(key, str) else key
                if safe_key in reduced:
                    raise ValueError(
                        "credential reduction would create duplicate object keys"
                    )
                normalized = normalize(str(key))
                property_schema = properties and isinstance(child, (Mapping, bool))
                child_secret = structured_secret or (
                    not property_schema
                    and (
                        is_sensitive(str(key))
                        or schema_secret
                        and normalized in SCHEMA_VALUES
                    )
                )
                reduced[safe_key] = self.reduce(
                    child,
                    child_secret,
                    is_sensitive(str(key)) if property_schema else schema_secret,
                    property_schema
                    or object_schema
                    or normalized in {"schema", "json_schema"},
                    not structured_secret
                    and object_schema
                    and normalized == "properties",
                )
            return reduced
        if isinstance(value, list):
            return [
                self.reduce(
                    child,
                    structured_secret,
                    schema_secret,
                    schema_context,
                    properties,
                )
                for child in value
            ]
        if isinstance(value, tuple):
            return tuple(
                self.reduce(
                    child,
                    structured_secret,
                    schema_secret,
                    schema_context,
                    properties,
                )
                for child in value
            )
        if (
            structured_secret
            and value is not None
            and not isinstance(value, bool)
            and (
                not isinstance(value, str)
                or value.strip()
                and not PLACEHOLDER.fullmatch(value.strip())
            )
        ):
            self.replacements += 1
            return REDACTED
        if not isinstance(value, str):
            return value
        return self.redact_text(value)

    def prepare(self, value: Any) -> tuple[Any, int]:
        self.discover(value)
        self.exact = (
            re.compile(
                "|".join(
                    re.escape(secret)
                    for secret in sorted(self.secrets, key=len, reverse=True)
                )
            )
            if self.secrets
            else None
        )
        reduced = self.reduce(value)
        count = self.replacements
        self.reduce(reduced)
        if self.replacements != count:
            raise ValueError("reduced upload still contains credentials")
        return reduced, count


def prepare_upload(
    value: Any,
    known_secrets: Iterable[str] = (),
    secret_sources: Iterable[Mapping[str, str]] = (),
    secret_fingerprints: Iterable[SecretFingerprint] = (),
) -> tuple[Any, int]:
    """Return a reduced copy and the number of credential-bearing values changed."""
    return CredentialReducer(
        known_secrets, secret_sources, secret_fingerprints
    ).prepare(value)
