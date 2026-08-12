"""Shared configuration for execution-time network policy."""

from fnmatch import fnmatchcase
from typing import Self
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_config import BaseConfig


def network_rule_matches(rule: str, scheme: str, host: str, port: int) -> bool:
    """Match a network-policy host pattern or URL origin. Paths are ignored."""
    value = rule.lower().rstrip("/")
    parsed = urlsplit(value if "://" in value else f"//{value}")
    pattern = (parsed.hostname or "").rstrip(".")
    if not pattern or (parsed.scheme and parsed.scheme != scheme):
        return False
    try:
        rule_port = parsed.port
    except ValueError:
        return False
    if parsed.scheme and rule_port is None:
        rule_port = 443 if parsed.scheme == "https" else 80
    if rule_port is not None and rule_port != port:
        return False
    host = host.lower().rstrip(".")
    return fnmatchcase(host, pattern) or (
        pattern.startswith("*.") and host == pattern[2:]
    )


class NetworkPolicyConfig(BaseConfig):
    """Shared execution-time policy surface for runtimes that support it."""

    allow: list[str] = Field(default_factory=lambda: ["*"])
    """Destinations allowed during execution; `*` is unrestricted and `[]` is
    framework-only."""
    block: list[str] = Field(default_factory=list)
    """Destinations denied during execution; any `*` makes the policy framework-only."""

    @model_validator(mode="after")
    def validate_network_policy(self) -> Self:
        if not self.allow or "*" in self.block:
            # Empty allowlists and wildcard blocks both mean framework-only access.
            self.allow = []
            self.block = ["*"]
        elif self.allow != ["*"] and self.block:
            raise ValueError(
                "non-empty concrete allow and block egress lists are mutually exclusive"
            )
        return self

    @property
    def network_restricted(self) -> bool:
        return "*" not in self.allow or bool(self.block)

    def permits(self, url: str) -> bool:
        """Whether this policy permits the destination named by an absolute URL."""
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        try:
            port = parsed.port
        except ValueError:
            return False
        if port is None:
            port = {"http": 80, "https": 443}.get(parsed.scheme)
        if port is None:
            return False
        if (
            parsed.scheme == "https"
            and port != 443
            and not any(
                rule == "*"
                or urlsplit(rule.lower()).scheme == "https"
                and network_rule_matches(rule, "https", parsed.hostname, port)
                for rule in self.allow
            )
        ):
            return False
        if any(
            network_rule_matches(rule, parsed.scheme, parsed.hostname, port)
            for rule in self.block
        ):
            return False
        return any(
            network_rule_matches(rule, parsed.scheme, parsed.hostname, port)
            for rule in self.allow
        )

    def with_task_network_policy(self, allow: list[str], block: list[str]) -> Self:
        values = self.model_dump()
        if not allow or not self.allow or "*" in block:
            # Framework-only access is absorbing; composition cannot widen either side.
            return type(self).model_validate({**values, "allow": [], "block": ["*"]})
        if "*" not in allow:
            allow = (
                allow
                if "*" in self.allow
                else list(dict.fromkeys([*allow, *self.allow]))
            )
        else:
            allow = self.allow
        block = list(dict.fromkeys([*block, *self.block]))
        return type(self).model_validate({**values, "allow": allow, "block": block})
