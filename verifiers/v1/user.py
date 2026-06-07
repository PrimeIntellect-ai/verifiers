from __future__ import annotations

from .toolset import Scope, ServerConfig, Toolset, load_toolset


class UserConfig(ServerConfig):
    name: str | None = "user"
    scope: Scope = "rollout"

    def load(self) -> "User":
        user = load_toolset(self.loader, self)
        if not isinstance(user, User):
            raise TypeError(f"User loader {self.loader!r} did not return a User.")
        return user


class User(Toolset[UserConfig]):
    pass
