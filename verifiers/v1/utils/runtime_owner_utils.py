from collections.abc import Iterable

from ..config import Config
from ..toolset import Toolset, merge_toolsets, normalize_toolset_collection
from ..types import Handler
from ..user import normalize_user
from .config_callable_utils import CallableKind, merge_config_handler_map


_HANDLER_KINDS: tuple[CallableKind, ...] = (
    "stop",
    "setup",
    "update",
    "metric",
    "reward",
    "advantage",
    "cleanup",
    "teardown",
)


class RuntimeOwnerMixin:
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

    def _init_runtime_user(self) -> None:
        self.user = normalize_user(getattr(self.config, "user"))

    def _init_runtime_toolsets(self) -> None:
        self.toolsets, self.named_toolsets = merge_toolsets(
            (), getattr(self.config, "toolsets")
        )

    def _init_runtime_handlers(self, *, base_metrics: Iterable[Handler] = ()) -> None:
        defaults: dict[CallableKind, Iterable[Handler]] = {
            kind: () for kind in _HANDLER_KINDS
        }
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
