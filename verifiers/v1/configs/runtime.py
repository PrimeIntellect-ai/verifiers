"""Runtime provider configs and the execution-time network policy they share."""

from fnmatch import fnmatchcase
from glob import has_magic
from itertools import product
from typing import Annotated, Any, Literal, Self
from urllib.parse import SplitResult, urlsplit

from pydantic import Field, SerializeAsAny, model_validator
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
    """What a box may reach. A runtime's own rules, a task's (`TaskData.network`) and a
    task's scoring (`ScoringConfig.network`) all use it; a task's policy combines with its
    runtime's rules: allowlists intersect and blocklists merge."""

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

    def with_task_network_policy(self, network: "NetworkPolicyConfig") -> Self:
        values = self.model_dump()
        # Intersection must not erase syntax the runtime cannot enforce.
        task = type(self).model_validate(
            {**values, "allow": network.allow, "block": network.block}
        )
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


class SubprocessConfig(BaseConfig):
    type: Literal["subprocess"] = "subprocess"


class DockerConfig(NetworkPolicyConfig):
    type: Literal["docker"] = "docker"
    image: str = "python:3.11-slim"
    workdir: str | None = None
    """Working directory override; None uses the task's workdir, or /app."""
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float | None = None
    """Pin the container to this many CPU cores. None = unlimited."""
    memory: float | None = None
    """Hard memory limit in GB. None = unlimited."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "2". Docker exposes that many GPUs (needs the nvidia
    container toolkit) and checks requested types with nvidia-smi. None = none."""
    disk: float | None = None
    """Advisory disk request in GB. Docker has no portable per-container size limit, so this is accepted (so a task can declare it without a warning) but not
    enforced."""


class PrimeConfig(NetworkPolicyConfig):
    type: Literal["prime"] = "prime"
    image: str = "python:3.11-slim"
    """Docker image to run. Any pullable ref works: on the first use of an image, the
    platform auto-builds the VM image that the sandbox needs from it (~10 minutes)
    and caches the result, so later sandboxes on the same ref start in
    seconds."""
    workdir: str | None = None
    """Working directory override; None uses the task's workdir, or /app."""
    region: str | None = None
    """Region to provision in (None = provider-chosen)."""
    labels: list[str] = Field(default_factory=list)
    """Labels attached to the sandbox, extending any process-wide base labels (see ``set_base_sandbox_labels``)."""
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float = 1.0
    """CPU cores."""
    memory: float = 2.0
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "A100:2" (a bare count = provider-chosen type)."""
    disk: float = 5.0
    """Disk in GB. Unused with ``checkpoint``: a sandbox keeps its checkpoint's disk."""
    checkpoint: str | None = None
    """A DURABLE filesystem checkpoint to start from instead of ``image`` (see the SDK's
    ``checkpoint`` and ``wait_for_checkpoint``)."""
    idle_timeout: float | None = 3600
    """Seconds of inactivity before the sandbox self-deletes (None disables)."""
    creates_per_min: int | None = None
    """Pace sandbox creation to this many per minute, enforced run-wide across every
    env-server worker process (None/<= 0 disables it). (Tunnel creation is limited separately
    — see interception.tunnel.prime.tunnel_limiter.)"""

    @model_validator(mode="after")
    def _validate_egress(self) -> Self:
        if not self.network_restricted:
            return self
        if not self.allow:
            return self
        from prime_sandboxes.models import validate_egress_lists

        validate_egress_lists(
            None if self.allow == ["*"] else self.allow,
            self.block or None,
        )
        return self


class CustomRuntimeConfig(BaseConfig):
    type: Literal["custom"] = "custom"
    import_path: str
    """Dotted path to a `RuntimeBackend` subclass, e.g. `my_package.runtime.MyBackend`."""

    @model_validator(mode="wrap")
    @classmethod
    def _validate_as_plugin_config(cls, data: Any, handler):
        """Validate the raw config as the backend's own `config_class`."""
        if (
            cls is not CustomRuntimeConfig
            or not isinstance(data, dict)
            or "import_path" not in data
        ):
            return handler(data)
        from verifiers.v1.runtimes.backends.custom import (
            import_backend,
            plugin_config_class,
        )

        config_class = plugin_config_class(import_backend(data["import_path"]))
        if config_class is CustomRuntimeConfig:
            return handler(data)
        return config_class.model_validate(data)


# Serialized by the runtime type, so a custom runtime's typed config keeps its fields.
RuntimeConfig = SerializeAsAny[
    Annotated[
        SubprocessConfig | DockerConfig | PrimeConfig | CustomRuntimeConfig,
        Field(discriminator="type"),
    ]
]
