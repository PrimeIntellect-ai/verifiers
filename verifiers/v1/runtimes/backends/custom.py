"""Custom runtimes: a `RuntimeBackend` subclass imported from your own package, so a
provider can be used without upstreaming it.

    [env.agent.runtime]
    type = "custom"
    import_path = "my_package.runtime.MyBackend"
    region = "us-east-1"

The backend sets `config_class` to its own `CustomRuntimeConfig` subclass, which
validates every field next to `import_path`. The config's bases declare what it
supports, as for the built-in runtimes: inherit `ImageConfig` to run task images and
`NetworkPolicyConfig` to enforce task network policies. The backend's class attributes
(`supports_user`, `supports_expose`, `supports_compose`, `is_local`) declare the rest.
The framework refuses features a runtime does not declare before provisioning."""

import importlib
from typing import Any, Literal

from pydantic import ConfigDict, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.runtimes.base import BaseRuntimeInfo, RuntimeBackend


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
        config_class = plugin_config_class(import_backend(data["import_path"]))
        if config_class is CustomRuntimeConfig:
            return handler(data)
        return config_class.model_validate(data)


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
