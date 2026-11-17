"""Custom runtimes: a `RuntimeBackend` subclass imported from your own package, so a
provider can be used without upstreaming it.

    [env.agent.runtime]
    type = "custom"
    import_path = "my_package.runtime.MyBackend"
    region = "us-east-1"

The backend sets `config_class` to its own `CustomRuntimeConfig` subclass, which
validates every field next to `import_path` and inherits the `SandboxConfig` fields
every runtime honors. The backend implements the full `RuntimeBackend` baseline, as
the built-in runtimes do; `is_local` and `SUPPORTS_DISK_LIMIT` declare its locality and
disk enforcement, and a config that inherits `NetworkPolicyConfig` enforces task network
policies. The framework refuses an undeclared network policy or disk limit before
provisioning."""

import importlib
from typing import Literal

from pydantic import ConfigDict

from verifiers.v1.configs.runtime import CustomRuntimeConfig
from verifiers.v1.runtimes.base import BaseRuntimeInfo, RuntimeBackend


class CustomRuntimeInfo(BaseRuntimeInfo):
    """The resolved config of a custom runtime, kept as plain data."""

    model_config = ConfigDict(extra="allow")
    type: Literal["custom"] = "custom"
    import_path: str


def import_backend(import_path: str) -> type[RuntimeBackend]:
    module_path, _, name = import_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"custom runtime import_path {import_path!r} must be a dotted path such as "
            "'my_package.runtime.MyBackend'"
        )
    try:
        backend = getattr(importlib.import_module(module_path), name)
    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"cannot import custom runtime {import_path!r}: {e}. Install the package "
            "that provides it in the environment running the rollouts."
        ) from e
    if not (isinstance(backend, type) and issubclass(backend, RuntimeBackend)):
        raise TypeError(
            f"custom runtime {import_path!r} must be a RuntimeBackend subclass, "
            f"not {backend!r}"
        )
    plugin_config_class(backend)
    return backend


def plugin_config_class(backend: type[RuntimeBackend]) -> type[CustomRuntimeConfig]:
    """The backend's typed config; `CustomRuntimeConfig` when it declares none."""
    config_class = getattr(backend, "config_class", CustomRuntimeConfig)
    if not (
        isinstance(config_class, type) and issubclass(config_class, CustomRuntimeConfig)
    ):
        raise TypeError(
            f"custom runtime {backend.__name__} must set config_class to a "
            "CustomRuntimeConfig subclass"
        )
    return config_class
