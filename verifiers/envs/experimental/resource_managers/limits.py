"""Centralized limits and timeout configuration for resource managers.

Inspired by frontier-evals' Alcatraz limits pattern - keeps all timeouts
and resource limits in one place for easy configuration and overrides.
"""

from typing import TypedDict


class SandboxLimits(TypedDict, total=False):
    """Centralized timeouts and limits for sandbox operations.

    All time values are in seconds. Use total=False to make all keys optional,
    allowing partial overrides of DEFAULT_SANDBOX_LIMITS.

    Example:
        # Override just the command timeout
        limits = {**DEFAULT_SANDBOX_LIMITS, "command_timeout_seconds": 120}

        # Or pass partial limits (merged with defaults internally)
        manager = SandboxManager(limits={"command_timeout_seconds": 120})
    """

    # Command execution
    command_timeout_seconds: int
    """Default timeout for individual commands (default: 30)"""

    health_check_timeout_seconds: int
    """Timeout for health check commands (default: 10)"""

    # Sandbox lifecycle
    creation_timeout_seconds: int
    """Max time to wait for sandbox creation API call (default: 60)"""

    ready_timeout_seconds: int
    """Max time to wait for sandbox to become ready (default: 300)"""

    ready_poll_interval_seconds: float
    """Interval between ready status checks (default: 1.0)"""

    ready_max_attempts: int
    """Max attempts to check if sandbox is ready (default: 120)"""

    # Background jobs
    background_job_default_timeout_seconds: int
    """Default timeout for background jobs (default: 3600)"""

    background_job_poll_interval_seconds: int
    """Interval between background job status checks (default: 3)"""

    # Health monitoring
    health_check_interval_seconds: float
    """Interval between health monitoring checks (default: 30.0)"""

    health_check_consecutive_failures: int
    """Number of consecutive failures before marking unhealthy (default: 3)"""

    # Cleanup
    destroy_timeout_seconds: int
    """Timeout for destroying a single sandbox (default: 30)"""

    bulk_delete_batch_size: int
    """Number of sandboxes to delete per batch in release_all (default: 100)"""

    # Garbage collection
    gc_interval_seconds: float
    """Interval between garbage collection runs (default: 60.0)"""

    gc_orphan_threshold_seconds: float
    """Time after which unreferenced sandbox is considered orphaned (default: 300.0)"""

    # Concurrency
    max_concurrent_creates: int
    """Max number of sandboxes being created simultaneously (default: 50)"""

    max_concurrent_commands: int
    """Max concurrent command executions (default: 100)"""

    # Client configuration
    client_max_workers: int
    """Thread pool size for async client wrapper (default: 100)"""

    client_max_connections: int
    """Max HTTP connections to sandbox API (default: 100)"""

    client_max_keepalive_connections: int
    """Max keepalive connections (default: 50)"""


DEFAULT_SANDBOX_LIMITS: SandboxLimits = {
    # Command execution
    "command_timeout_seconds": 30,
    "health_check_timeout_seconds": 10,

    # Sandbox lifecycle
    "creation_timeout_seconds": 60,
    "ready_timeout_seconds": 300,
    "ready_poll_interval_seconds": 1.0,
    "ready_max_attempts": 120,

    # Background jobs
    "background_job_default_timeout_seconds": 3600,
    "background_job_poll_interval_seconds": 3,

    # Health monitoring
    "health_check_interval_seconds": 30.0,
    "health_check_consecutive_failures": 3,

    # Cleanup
    "destroy_timeout_seconds": 30,
    "bulk_delete_batch_size": 100,

    # Garbage collection
    "gc_interval_seconds": 60.0,
    "gc_orphan_threshold_seconds": 300.0,

    # Concurrency
    "max_concurrent_creates": 50,
    "max_concurrent_commands": 100,

    # Client configuration
    "client_max_workers": 100,
    "client_max_connections": 100,
    "client_max_keepalive_connections": 50,
}


def merge_limits(base: SandboxLimits, overrides: SandboxLimits | None) -> SandboxLimits:
    """Merge override limits into base limits.

    Args:
        base: Base limits (typically DEFAULT_SANDBOX_LIMITS)
        overrides: Optional partial limits to override

    Returns:
        Merged limits with overrides taking precedence
    """
    if overrides is None:
        return base
    return {**base, **overrides}  # type: ignore[return-value]
