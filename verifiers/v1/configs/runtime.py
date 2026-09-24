"""Shared execution runtime configuration."""

from fnmatch import fnmatchcase
from glob import has_magic
from itertools import product
from typing import Self
from urllib.parse import SplitResult, urlsplit

from pydantic import Field, model_validator
from pydantic_config import BaseConfig


class BindMount(BaseConfig):
    """An existing file or directory on the container engine's host."""

    source: str = Field(pattern=r"^/[^\x00]*$")
    """Absolute host path. The runtime never creates or removes the source."""
    read_only: bool = True


def parse_network_rule(rule: str) -> tuple[SplitResult, str, int | None]:
    """Normalize host/port while retaining URL syntax for backend validation."""
    parsed = urlsplit(rule if "://" in rule else f"//{rule}")
    port = parsed.port
    if parsed.scheme and port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed, (parsed.hostname or "").lower().rstrip("."), port


def network_rule_matches(rule: str, scheme: str, host: str, port: int) -> bool:
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
    return fnmatchcase(host, pattern) or (
        pattern.startswith("*.") and host == pattern[2:]
    )


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
        if fnmatchcase(child, parent) or child == parent.removeprefix("*."):
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

    def permits(self, scheme: str, host: str, port: int) -> bool:
        """Whether the destination is allowed by the configured egress rules."""
        return not any(
            network_rule_matches(rule, scheme, host, port) for rule in self.block
        ) and any(network_rule_matches(rule, scheme, host, port) for rule in self.allow)

    def with_task_network_policy(self, allow: list[str], block: list[str]) -> Self:
        values = self.model_dump()
        # Intersection must not erase syntax the runtime cannot enforce.
        task = type(self).model_validate({**values, "allow": allow, "block": block})
        allow, block = task.allow, task.block
        if "*" in allow:
            allow = self.allow
        elif "*" not in self.allow:
            intersection = []
            for left, right in product(allow, self.allow):
                if left == right:
                    intersection.append(left)
                    continue
                a, a_host, a_port = parse_network_rule(left)
                b, b_host, b_port = parse_network_rule(right)
                # Bare paths can be provider-specific CIDRs, not URL paths to discard.
                if any(
                    not url.hostname or (url.path and not url.scheme) for url in (a, b)
                ):
                    raise ValueError(
                        f"cannot intersect network rules {left!r} and {right!r}"
                    )
                if (
                    len({a.scheme, b.scheme} - {""}) > 1
                    or len({a_port, b_port} - {None}) > 1
                ):
                    continue
                host = intersect_network_hosts(a_host, b_host)
                if not host:
                    continue
                scheme = a.scheme or b.scheme
                port = a_port if a_port is not None else b_port
                authority = f"[{host}]" if ":" in host else host
                if port is not None:
                    authority = f"{authority}:{port}"
                intersection.append(f"{scheme}://{authority}" if scheme else authority)
            allow = list(dict.fromkeys(intersection))
        block = list(dict.fromkeys([*block, *self.block]))
        return type(self).model_validate({**values, "allow": allow, "block": block})
