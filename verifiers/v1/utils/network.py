"""Network rule matching shared by runtimes and standalone harnesses."""

from fnmatch import fnmatchcase
from urllib.parse import SplitResult, urlsplit

from pydantic import AnyHttpUrl


def normalize_network_host(host: str) -> str:
    """Canonicalize IDNs while preserving ASCII host patterns."""
    host = host.lower().rstrip(".")
    if not host.isascii():
        host = AnyHttpUrl(f"http://{host}").host.strip("[]").rstrip(".")
    return host


def parse_network_rule(rule: str) -> tuple[SplitResult, str, int | None]:
    """Normalize host/port while retaining URL syntax for backend validation."""
    parsed = urlsplit(rule if "://" in rule else f"//{rule}")
    port = parsed.port
    if parsed.scheme and port is None:
        port = 443 if parsed.scheme == "https" else 80
    return parsed, normalize_network_host(parsed.hostname or ""), port


def network_rule_matches(
    rule: str, scheme: str, host: str, port: int, *, subdomains: bool = False
) -> bool:
    """Match a network-policy host pattern or URL origin. Paths are ignored."""
    try:
        parsed, pattern, rule_port = parse_network_rule(rule)
        host = normalize_network_host(host)
    except ValueError:
        return False
    if not pattern or (parsed.scheme and parsed.scheme != scheme):
        return False
    if rule_port is not None and rule_port != port:
        return False
    return fnmatchcase(host, pattern) or (subdomains and host.endswith(f".{pattern}"))
