from collections.abc import Iterable

from ..config import LifecycleConfig
from ..toolset import Toolset, Toolsets, collect_toolsets, normalize_toolset_collection
from ..types import Handler
from ..user import UserConfig, normalize_user
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
    config: LifecycleConfig
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

    def load_user(self) -> UserConfig | None:
        return None

    def load_toolsets(self) -> Toolsets:
        return None

    def initialize_runtime_user(
        self, user: UserConfig | None, *, explicitly_configured: bool
    ) -> None:
        if explicitly_configured:
            self.user = normalize_user(user)
            return
        self.user = normalize_user(self.load_user())

    def initialize_runtime_toolsets(self, toolsets: object) -> None:
        if toolsets is None:
            self.toolsets = []
            self.named_toolsets = {}
            return
        self.toolsets, self.named_toolsets = collect_toolsets(
            self.load_toolsets(), toolsets
        )

    def initialize_runtime_handlers(self) -> None:
        defaults: dict[CallableKind, Iterable[Handler]] = {
            kind: () for kind in _HANDLER_KINDS
        }
        handlers = merge_config_handler_map(defaults, self.config)
        self.stops = handlers["stop"]
        self.setups = handlers["setup"]
        self.updates = handlers["update"]
        self.metrics = handlers["metric"]
        self.rewards = handlers["reward"]
        self.advantages = handlers["advantage"]
        self.cleanups = handlers["cleanup"]
        self.teardowns = handlers["teardown"]

    def refresh_runtime(self) -> None:
        pass

    def add_metric(self, fn: Handler) -> None:
        self.metrics.append(fn)
        self.refresh_runtime()

    def add_reward(self, fn: Handler) -> None:
        self.rewards.append(fn)
        self.refresh_runtime()

    def add_advantage(self, fn: Handler) -> None:
        self.advantages.append(fn)
        self.refresh_runtime()

    def add_toolset(self, toolset: object) -> None:
        toolsets, named_toolsets = normalize_toolset_collection(toolset)
        duplicate = set(self.named_toolsets) & set(named_toolsets)
        if duplicate:
            raise ValueError(f"Toolsets are defined twice: {sorted(duplicate)}.")
        self.toolsets.extend(toolsets)
        self.named_toolsets.update(named_toolsets)
        self.refresh_runtime()

    def add_stop(self, fn: Handler) -> None:
        self.stops.append(fn)
        self.refresh_runtime()

    def add_setup(self, fn: Handler) -> None:
        self.setups.append(fn)
        self.refresh_runtime()

    def add_update(self, fn: Handler) -> None:
        self.updates.append(fn)
        self.refresh_runtime()

    def add_cleanup(self, fn: Handler) -> None:
        self.cleanups.append(fn)
        self.refresh_runtime()

    def add_teardown(self, fn: Handler) -> None:
        self.teardowns.append(fn)
        self.refresh_runtime()
