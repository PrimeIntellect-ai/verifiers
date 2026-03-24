"""Core transition semantics for depgraph preflight/search."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from .types import (
    ActionContract,
    BindingSourceSpec,
    SyncRuleSpec,
    WorldEffectSpec,
    WorldPredicateSpec,
)

_MAX_SYNC_ITERATIONS = 32


def leaf_field_name(path: str) -> str:
    """Derive a set_/get_/assert_ method suffix from a projected world path."""
    parts = path.split(".")
    if not parts or not parts[-1]:
        raise ValueError(f"Cannot derive field name from path '{path}'")
    if len(parts) >= 3 and "[" not in path:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return ("json", json.dumps(value, sort_keys=True))


def world_state_key(world: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return a hashable, deterministic key for projected world assignments."""
    return tuple(
        sorted((path, _normalize_value(value)) for path, value in world.items())
    )


def world_from_effects(
    effects: list[WorldEffectSpec],
) -> tuple[dict[str, Any], list[str]]:
    """Build world assignment map from effects and return conflict diagnostics."""
    world: dict[str, Any] = {}
    issues: list[str] = []
    for effect in effects:
        if effect.path in world and world[effect.path] != effect.set:
            issues.append(
                f"Conflicting assignments for path '{effect.path}': "
                f"{world[effect.path]!r} vs {effect.set!r}"
            )
            continue
        world[effect.path] = effect.set
    return world, issues


def predicate_holds(predicate: WorldPredicateSpec, world: dict[str, Any]) -> bool:
    value = world.get(predicate.path, None)
    if predicate.op == "eq":
        return value == predicate.value
    if predicate.op == "neq":
        return value != predicate.value
    if value is None:
        return False
    if predicate.op == "gt":
        return value > predicate.value
    if predicate.op == "lt":
        return value < predicate.value
    if predicate.op == "gte":
        return value >= predicate.value
    if predicate.op == "lte":
        return value <= predicate.value
    return False


def apply_sync_rules(
    sync_rules: list[SyncRuleSpec],
    world: dict[str, Any],
) -> dict[str, Any]:
    """Apply all eligible sync rules until the world reaches a fixed-point."""
    if not sync_rules:
        return world

    for _ in range(_MAX_SYNC_ITERATIONS):
        before_pass = world_state_key(world)
        for rule in sync_rules:
            if not all(
                predicate_holds(predicate, world) for predicate in rule.requires_world
            ):
                continue
            for effect in rule.effects_world:
                new_value = effect.set
                if effect.from_path is not None:
                    new_value = world.get(effect.from_path)
                if world.get(effect.path) != new_value:
                    world[effect.path] = new_value
        if world_state_key(world) == before_pass:
            return world

    raise ValueError(
        f"Sync rules did not converge after {_MAX_SYNC_ITERATIONS} iterations. "
        "Check for cyclical dependencies in sync_rules."
    )


def materialize_world(
    effects: list[WorldEffectSpec],
    *,
    sync_rules: list[SyncRuleSpec] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build a world map from literal effects and then normalize it through sync rules."""
    world, issues = world_from_effects(effects)
    if issues:
        return world, issues
    if sync_rules:
        world = apply_sync_rules(sync_rules, dict(world))
    return world, issues


def index_binding_sources(
    bindings: list[BindingSourceSpec],
) -> dict[str, list[BindingSourceSpec]]:
    """Index binding sources by produced binding id."""
    by_binding: dict[str, list[BindingSourceSpec]] = defaultdict(list)
    for source in bindings:
        by_binding[source.binding_id].append(source)
    return dict(by_binding)


def binding_source_satisfiable(
    source: BindingSourceSpec, world: dict[str, Any]
) -> bool:
    """Return True if source observability predicates are satisfied in the world state."""
    all_ok = all(
        predicate_holds(predicate, world) for predicate in source.observability_all_of
    )
    any_clause = source.observability_any_of
    any_ok = (
        True
        if not any_clause
        else any(predicate_holds(predicate, world) for predicate in any_clause)
    )
    return all_ok and any_ok


def _knowledge_source_ok(
    action: ActionContract,
    world: dict[str, Any],
    bindings: frozenset[str],
    binding_sources_by_id: dict[str, list[BindingSourceSpec]],
) -> bool:
    if action.classification != "knowledge-only":
        return True
    if not action.effects_bindings:
        return False
    for binding_id in action.effects_bindings:
        if binding_id in bindings:
            return False
        sources = binding_sources_by_id.get(binding_id, [])
        if not sources:
            return False
        tool_scoped_sources = [
            source
            for source in sources
            if source.source_tool == action.resolved_tool_name
        ]
        candidates = tool_scoped_sources if tool_scoped_sources else sources
        if not any(binding_source_satisfiable(source, world) for source in candidates):
            return False
    return True


def is_action_enabled(
    action: ActionContract,
    world: dict[str, Any],
    bindings: frozenset[str],
    binding_sources_by_id: dict[str, list[BindingSourceSpec]],
) -> bool:
    """Eligibility predicate for fan-out over (world, bindings)."""
    if not all(
        predicate_holds(predicate, world) for predicate in action.requires_world
    ):
        return False
    for predicate in action.requires_bindings:
        binding_present = predicate.binding_id in bindings
        if predicate.acquired and not binding_present:
            return False
        if not predicate.acquired and binding_present:
            return False
    if not _knowledge_source_ok(action, world, bindings, binding_sources_by_id):
        return False
    return True


def _invalidate_bindings(
    old_world: dict[str, Any],
    new_world: dict[str, Any],
    bindings: set[str],
    binding_specs: list[BindingSourceSpec],
) -> set[str]:
    """Drop acquired bindings whose canonical backing field changed value."""
    for spec in binding_specs:
        if spec.world_path is None:
            continue
        if spec.binding_id not in bindings:
            continue
        if old_world.get(spec.world_path) != new_world.get(spec.world_path):
            bindings.discard(spec.binding_id)
    return bindings


def apply_action(
    action: ActionContract,
    world: dict[str, Any],
    bindings: frozenset[str],
    *,
    sync_rules: list[SyncRuleSpec] | None = None,
    binding_specs: list[BindingSourceSpec] | None = None,
) -> tuple[dict[str, Any], frozenset[str]]:
    """Apply one action, then sync rules, then binding invalidation."""
    next_world = dict(world)
    for effect in action.effects_world:
        next_world[effect.path] = effect.set

    if sync_rules:
        next_world = apply_sync_rules(sync_rules, next_world)

    next_bindings = set(bindings)
    next_bindings.update(action.effects_bindings)
    if binding_specs:
        next_bindings = _invalidate_bindings(
            world, next_world, next_bindings, binding_specs
        )
    return next_world, frozenset(next_bindings)
