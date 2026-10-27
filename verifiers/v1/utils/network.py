"""Network rule types and matching shared by runtimes and standalone harnesses."""

from fnmatch import fnmatchcase
from typing import TypeAlias
from urllib.parse import SplitResult, urlsplit

NetworkAllowlist: TypeAlias = list[str]
NetworkBlocklist: TypeAlias = list[str]


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
