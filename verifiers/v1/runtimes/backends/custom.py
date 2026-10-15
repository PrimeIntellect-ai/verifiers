"""Custom runtimes: a `RuntimeBackend` subclass imported from your own package, so a
provider can be used without upstreaming it.

    [env.agent.runtime]
    type = "custom"
    import_path = "my_package.runtime.MyBackend"
    kwargs = { region = "us-east-1" }

The class is constructed as `MyBackend(config, name)` with this `CustomRuntimeConfig`,
like the built-in backends. It declares its capabilities (`supports_image`,
`supports_user`, `supports_expose`, `is_local`) as class attributes; the framework
refuses features a backend does not declare before provisioning anything."""

import importlib
from typing import Any, Literal

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import BaseRuntimeInfo, RuntimeBackend


class CustomRuntimeConfig(BaseConfig):
    type: Literal["custom"] = "custom"
    import_path: str
    """Dotted path to a `RuntimeBackend` subclass, e.g. `my_package.runtime.MyBackend`."""
    kwargs: dict[str, Any] = Field(default_factory=dict)
    """Provider settings, read by the backend from `config.kwargs`."""
    image: str | None = None
    """Task image, set from task data on backends with `supports_image`."""
    workdir: str | None = None
    """Working directory override; None uses the task's workdir. A started backend
    sets `config.workdir` to the directory its commands run in."""


class CustomRuntimeInfo(CustomRuntimeConfig, BaseRuntimeInfo):
    pass


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
    return backend
