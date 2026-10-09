"""Shared configuration for execution-time network policy."""

from fnmatch import fnmatchcase
from glob import has_magic
from typing import Self
from urllib.parse import SplitResult, urlsplit

from pydantic import Field, ValidationInfo, model_validator
from pydantic_config import BaseConfig


def parse_network_rule(rule: str) -> tuple[SplitResult, str, int | None]:
    """Normalize host/port while retaining URL syntax for backend validation."""
    parsed = urlsplit(rule if "://" in rule else f"//{rule}")
    port = parsed.port
    if parsed.scheme and port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed, (parsed.hostname or "").lower().rstrip("."), port


def network_rule_matches(
    rule: str, scheme: str, host: str, port: int, *, subdomains: bool = False
) -> bool:
    """Match a network-policy host pattern or URL origin. Paths are ignored."""
    try:
        parsed, pattern, rule_port = parse_network_rule(rule)
    except ValueError:
        return False
    if not pattern or (parsed.scheme and parsed.scheme != scheme):
        return False
    if rule_port is not None and rule_port != port:
        return False
    host = host.lower().rstrip(".")
    return fnmatchcase(host, pattern) or (subdomains and host.endswith(f".{pattern}"))


def intersect_network_hosts(left: str, right: str) -> str:
    """Return the narrower host pattern, rejecting unprovable glob overlaps."""
    for parent, child in ((left, right), (right, left)):
        if parent in ("*", child):
            return child
        # Matching a pattern as text proves containment only for domain suffixes.
        if has_magic(child) and not (
            parent.startswith("*.") and not has_magic(parent[2:])
        ):
            continue
        if fnmatchcase(child, parent):
            return child
    if all(has_magic(host) for host in (left, right)) and any(
        has_magic(host.removeprefix("*.")) for host in (left, right)
    ):
        raise ValueError(f"cannot intersect network hosts {left!r} and {right!r}")
    return ""


class NetworkPolicyConfig(BaseConfig):
    """Shared execution-time policy surface for runtimes that support it."""

    allow: list[str] = Field(default_factory=lambda: ["*"])
    """Destinations allowed during execution; `*` is unrestricted and `[]` is
    framework-only."""
    block: list[str] = Field(default_factory=list)
    """Denied destinations; bare domains include subdomains, `*.` excludes the base
    domain, and any `*` makes the policy framework-only."""

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

    def permits(self, scheme: str, host: str, port: int) -> bool:
        """Whether the destination is allowed by the configured egress rules."""
        return not any(
            network_rule_matches(rule, scheme, host, port, subdomains=True)
            for rule in self.block
        ) and any(network_rule_matches(rule, scheme, host, port) for rule in self.allow)


class EnforcedNetworkPolicy(NetworkPolicyConfig):
    """What a runtime that enforces egress carries: the task's resolved policy
    (`Task.network`), written by `with_network` at resolution. Not a config: a
    runtime has no policy of its own, so `allow`/`block` are refused from TOML/CLI
    and left out of dumps."""

    allow: list[str] = Field(default_factory=lambda: ["*"], exclude=True)
    block: list[str] = Field(default_factory=list, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _refuse_configured_policy(cls, data, info: ValidationInfo):
        if (
            isinstance(data, dict)
            and (data.keys() & {"allow", "block"})
            and not (info.context or {}).get("resolved")
        ):
            raise ValueError(
                "a runtime enforces the task's network policy; set it on the task "
                "(TaskData.network) or override it with [env.taskset.task.network], "
                "not on the runtime"
            )
        return data

    @property
    def network(self) -> NetworkPolicyConfig:
        return NetworkPolicyConfig(allow=self.allow, block=self.block)

    def with_network(self, policy: NetworkPolicyConfig) -> Self:
        """This runtime config enforcing `policy`; the runtime's own validators
        reject rules it cannot express."""
        return type(self).model_validate(
            {**self.model_dump(), "allow": policy.allow, "block": policy.block},
            context={"resolved": True},
        )
