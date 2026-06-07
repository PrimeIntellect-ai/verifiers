from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Generic, TypeVar

from .toolset import (
    Scope,
    ServerConfig,
    Toolset,
    tool,
)


class UserConfig(ServerConfig):
    scope: Scope = "rollout"

    def default_server_ref(self) -> str:
        config_type = type(self)
        module_name = config_type.__module__
        if module_name.startswith("verifiers.v1."):
            raise ValueError(
                f"{config_type.__name__} cannot infer a user implementation from "
                "the framework package."
            )
        if module_name.endswith(".config"):
            package = type(self).config_package(module_name)
            if config_type.__name__ == "UserConfig":
                impl_name = "User"
            elif config_type.__name__.endswith("Config"):
                impl_name = config_type.__name__.removesuffix("Config")
            else:
                impl_name = "User"
            return f"{package}.user:{impl_name}"
        if config_type.__name__.endswith("Config"):
            ref = f"{module_name}:{config_type.__name__.removesuffix('Config')}"
        else:
            ref = f"{module_name}:{config_type.__name__}User"
        return type(self).resolve_ref(ref, config_type)

    def load(self) -> "User":
        server = self.implementation_ref()
        user = Toolset.load_ref(server, self)
        if not isinstance(user, User):
            raise TypeError(f"User server {server!r} did not return a User.")
        return user


UserConfigT = TypeVar("UserConfigT", bound=UserConfig)


class User(Toolset[UserConfigT], Generic[UserConfigT]):
    pass


UserFunc = TypeVar("UserFunc", bound=Callable[..., object])


def user(
    func: UserFunc | None = None,
    *,
    args: Mapping[str, str] | None = None,
    sets: Mapping[str, str] | None = None,
    extends: Mapping[str, str] | None = None,
) -> UserFunc | Callable[[UserFunc], UserFunc]:
    return tool(
        func, args=args, sets=sets, extends=extends, name="respond", hidden=True
    )
