from typing import Any, cast

from verifiers.types import RoutedExperts


def parse_routed_experts(raw: Any) -> RoutedExperts | None:
    if raw is None:
        return None
    return cast(RoutedExperts, raw)
