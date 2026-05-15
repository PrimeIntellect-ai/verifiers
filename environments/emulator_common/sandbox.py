import os

from verifiers.v1.types import ConfigData

WORKSPACE = "/workspace"
DEFAULT_TOOLCHAIN_IMAGE = "primeintellect/programbench-toolchain:latest"
DEFAULT_CPU_CORES = 4
DEFAULT_MEMORY_GB = 8
DEFAULT_DISK_GB = 12
DEFAULT_TIMEOUT_MINUTES = 360


def toolchain_image() -> str:
    return os.environ.get("PRIME_TOOLCHAIN_IMAGE", DEFAULT_TOOLCHAIN_IMAGE)


def build_sandbox_config(
    *,
    cpu_cores: int | None = None,
    memory_gb: int | None = None,
    network_access: bool = True,
    timeout_minutes: int | None = None,
) -> ConfigData:
    resolved_timeout = timeout_minutes or DEFAULT_TIMEOUT_MINUTES
    return {
        "image": toolchain_image(),
        "cpu_cores": cpu_cores or DEFAULT_CPU_CORES,
        "memory_gb": memory_gb or DEFAULT_MEMORY_GB,
        "disk_size_gb": DEFAULT_DISK_GB,
        "gpu_count": 0,
        "workdir": WORKSPACE,
        "scope": "rollout",
        "timeout_minutes": resolved_timeout,
        "command_timeout": resolved_timeout * 60,
        # Required for the vf-eval model tunnel. The benchmark itself does not
        # require source lookup network access.
        "network_access": network_access,
    }
