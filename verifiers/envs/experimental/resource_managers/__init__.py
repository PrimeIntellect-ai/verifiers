"""Resource management abstractions for sandboxes."""

from verifiers.envs.experimental.resource_managers.base import (
    ManagedResource,
    ResourceManager,
    ResourceState,
)
from verifiers.envs.experimental.resource_managers.errors import (
    CommandTimeoutError,
    SandboxCreationError,
    SandboxError,
    SandboxExecutionError,
    SandboxFailureInfo,
    SandboxNotReadyError,
    SandboxOOMError,
    SandboxSetupError,
    SandboxTimeoutError,
)
from verifiers.envs.experimental.resource_managers.limits import (
    DEFAULT_SANDBOX_LIMITS,
    SandboxLimits,
    merge_limits,
)
from verifiers.envs.experimental.resource_managers.recorder import (
    CommandEvent,
    InMemoryRecorder,
    NullRecorder,
    Recorder,
)
from verifiers.envs.experimental.resource_managers.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
)
from verifiers.envs.experimental.resource_managers.sandbox_manager import (
    BackgroundJob,
    ManagedSandbox,
    SandboxManager,
    sandbox_manager,
)

__all__ = [
    # Base
    "ManagedResource",
    "ResourceManager",
    "ResourceState",
    # Errors
    "CommandTimeoutError",
    "SandboxCreationError",
    "SandboxError",
    "SandboxExecutionError",
    "SandboxFailureInfo",
    "SandboxNotReadyError",
    "SandboxOOMError",
    "SandboxSetupError",
    "SandboxTimeoutError",
    # Limits
    "DEFAULT_SANDBOX_LIMITS",
    "SandboxLimits",
    "merge_limits",
    # Recorder
    "CommandEvent",
    "InMemoryRecorder",
    "NullRecorder",
    "Recorder",
    # Retry
    "DEFAULT_RETRY_CONFIG",
    "RetryConfig",
    # Sandbox Manager
    "BackgroundJob",
    "ManagedSandbox",
    "SandboxManager",
    "sandbox_manager",
]
