from collections.abc import Iterable
from typing import ClassVar

from ..config import Config, ConfigSource
from ..toolset import Toolset, merge_toolsets, normalize_toolset_collection
from ..types import Handler
from ..user import normalize_user
from .config_callable_utils import CallableKind, merge_config_handler_map


_HANDLER_DEFAULTS: dict[CallableKind, tuple[str, str]] = {
    "stop": ("stops", "_default_stops"),
    "setup": ("setups", "_default_setups"),
    "update": ("updates", "_default_updates"),
    "metric": ("metrics", "_default_metrics"),
    "reward": ("rewards", "_default_rewards"),
    "advantage": ("advantages", "_default_advantages"),
    "cleanup": ("cleanups", "_default_cleanups"),
    "teardown": ("teardowns", "_default_teardowns"),
}


class RuntimeOwnerMixin:
    _config_cls: ClassVar[type[Config]]
    _default_user: ClassVar[object | None] = None
    _default_toolsets: ClassVar[object] = ()
    _default_stops: ClassVar[tuple[Handler, ...]] = ()
    _default_setups: ClassVar[tuple[Handler, ...]] = ()
    _default_updates: ClassVar[tuple[Handler, ...]] = ()
    _default_metrics: ClassVar[tuple[Handler, ...]] = ()
    _default_rewards: ClassVar[tuple[Handler, ...]] = ()
    _default_advantages: ClassVar[tuple[Handler, ...]] = ()
    _default_cleanups: ClassVar[tuple[Handler, ...]] = ()
    _default_teardowns: ClassVar[tuple[Handler, ...]] = ()

    config: Config
    toolsets: list[Toolset]
    named_toolsets: dict[str, Toolset]
    stops: list[Handler]
    setups: list[Handler]
    updates: list[Handler]
    metrics: list[Handler]
    rewards: list[Handler]
    advantages: list[Handler]
    cleanups: list[Handler]
    teardowns: list[Handler]

    def _coerce_config(self, config: ConfigSource = None) -> Config:
        return type(self)._config_cls.from_config(config)

    def _field_was_set(self, field: str) -> bool:
        return field in self.config.model_fields_set

    def _defaulted(self, field: str, class_default: object) -> object:
        value = getattr(self.config, field)
        if value is None and not self._field_was_set(field):
            return class_default
        return value

    def _init_runtime_user(self) -> None:
        self.user = normalize_user(self._defaulted("user", type(self)._default_user))

    def _init_runtime_toolsets(self) -> None:
        default_toolsets = (
            () if self._field_was_set("toolsets") else type(self)._default_toolsets
        )
        self.toolsets, self.named_toolsets = merge_toolsets(
            default_toolsets, getattr(self.config, "toolsets")
        )

    def _init_runtime_handlers(self, *, base_metrics: Iterable[Handler] = ()) -> None:
        defaults: dict[CallableKind, Iterable[Handler]] = {}
        for kind, (field, attr) in _HANDLER_DEFAULTS.items():
            class_default = getattr(type(self), attr)
            defaults[kind] = () if self._field_was_set(field) else class_default
        defaults["metric"] = [*base_metrics, *defaults["metric"]]
        handlers = merge_config_handler_map(defaults, self.config)
        self.stops = handlers["stop"]
        self.setups = handlers["setup"]
        self.updates = handlers["update"]
        self.metrics = handlers["metric"]
        self.rewards = handlers["reward"]
        self.advantages = handlers["advantage"]
        self.cleanups = handlers["cleanup"]
        self.teardowns = handlers["teardown"]

    def _configure_from_config(self) -> None:
        pass

    def _runtime_owner_changed(self) -> None:
        pass

    def _add_handler(self, handlers: list[Handler], fn: Handler) -> None:
        handlers.append(fn)
        self._runtime_owner_changed()

    def add_metric(self, fn: Handler) -> None:
        self._add_handler(self.metrics, fn)

    def add_reward(self, fn: Handler) -> None:
        self._add_handler(self.rewards, fn)

    def add_advantage(self, fn: Handler) -> None:
        self._add_handler(self.advantages, fn)

    def add_toolset(self, toolset: object) -> None:
        toolsets, named_toolsets = normalize_toolset_collection(toolset)
        duplicate = set(self.named_toolsets) & set(named_toolsets)
        if duplicate:
            raise ValueError(f"Toolsets are defined twice: {sorted(duplicate)}.")
        self.toolsets.extend(toolsets)
        self.named_toolsets.update(named_toolsets)
        self._runtime_owner_changed()

    def add_stop(self, fn: Handler) -> None:
        self._add_handler(self.stops, fn)

    def add_setup(self, fn: Handler) -> None:
        self._add_handler(self.setups, fn)

    def add_update(self, fn: Handler) -> None:
        self._add_handler(self.updates, fn)

    def add_cleanup(self, fn: Handler) -> None:
        self._add_handler(self.cleanups, fn)

    def add_teardown(self, fn: Handler) -> None:
        self._add_handler(self.teardowns, fn)
