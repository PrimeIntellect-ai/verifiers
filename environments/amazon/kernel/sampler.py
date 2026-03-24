"""Fan-out sampler for generating candidate depgraph task intents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .preflight import (
    SearchResult,
    TaskPreflightReport,
    run_task_preflight,
    terminal_profile_map,
    world_matches_terminal_profile,
)
from .semantics import (
    apply_action,
    index_binding_sources,
    is_action_enabled,
    materialize_world,
    world_state_key,
)
from .types import (
    GraphContractSpec,
    SamplingRequestDoc,
    TaskIntent,
    TaskSpecsDoc,
)


@dataclass
class _Node:
    world: dict[str, Any]
    bindings: frozenset[str]
    plan: list[str]


@dataclass
class SampledTask:
    """Sampled task plus preflight report."""

    task: TaskIntent
    report: TaskPreflightReport


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def sample_task_intents(
    contract: GraphContractSpec,
    request: SamplingRequestDoc,
) -> list[SampledTask]:
    """Generate candidate task intents by BFS fan-out.

    Task identity is (seed_id, terminal_profile_id). Different BFS paths
    from the same seed to the same terminal profile collapse to one task
    (first/shortest hit wins). goal_world is populated directly from the
    matched terminal profile's requires_world predicates.
    """
    sampled: list[SampledTask] = []
    binding_sources_by_id = index_binding_sources(contract.bindings)
    terminal_profiles_by_id = terminal_profile_map(request.terminal_profiles)
    task_counter = 0
    n_seeds = len(request.seeds)
    per_seed_budget = (
        max(1, request.max_tasks // n_seeds) if n_seeds > 0 else request.max_tasks
    )

    for seed in request.seeds:
        seed_count = 0
        allowed_terminal_profiles = [
            terminal_profiles_by_id[profile_id]
            for profile_id in seed.allowed_terminal_profiles
        ]
        seen_profiles: set[str] = set()

        start_world, seed_issues = materialize_world(
            seed.start_world,
            sync_rules=contract.sync_rules,
        )
        if seed_issues:
            continue
        start_bindings = frozenset(seed.start_bindings)

        root = _Node(world=start_world, bindings=start_bindings, plan=[])
        queue: deque[_Node] = deque([root])
        visited: set[tuple[tuple[tuple[str, Any], ...], frozenset[str]]] = {
            (world_state_key(root.world), root.bindings)
        }

        while queue and seed_count < per_seed_budget:
            node = queue.popleft()
            if len(node.plan) >= seed.max_depth:
                continue

            for action in contract.actions:
                if not is_action_enabled(
                    action, node.world, node.bindings, binding_sources_by_id
                ):
                    continue
                next_world, next_bindings = apply_action(
                    action,
                    node.world,
                    node.bindings,
                    sync_rules=contract.sync_rules,
                    binding_specs=contract.bindings,
                )
                if next_world == node.world and next_bindings == node.bindings:
                    continue
                next_plan = node.plan + [action.action_id]
                state_key = (world_state_key(next_world), next_bindings)
                if state_key in visited:
                    continue
                visited.add(state_key)

                matched_terminal_profile = next(
                    (
                        profile
                        for profile in allowed_terminal_profiles
                        if world_matches_terminal_profile(next_world, profile)
                    ),
                    None,
                )

                depth = len(next_plan)
                if matched_terminal_profile is None or depth < seed.min_depth:
                    queue.append(
                        _Node(world=next_world, bindings=next_bindings, plan=next_plan)
                    )
                    continue

                if matched_terminal_profile.profile_id in seen_profiles:
                    continue
                seen_profiles.add(matched_terminal_profile.profile_id)

                goal_world = list(matched_terminal_profile.requires_world)

                required_actions = _ordered_unique(next_plan)
                task_id = f"{seed.seed_id}_d{depth}_{task_counter:03d}"
                task_counter += 1

                candidate = TaskIntent(
                    task_id=task_id,
                    start_world=seed.start_world,
                    start_bindings=seed.start_bindings,
                    goal_world=goal_world,
                    terminal_profile_id=matched_terminal_profile.profile_id,
                    required_actions=required_actions,
                    min_plan_length=depth,
                    runtime=None,
                )
                bfs_sat = SearchResult(
                    sat=True,
                    plan=next_plan,
                    explored_states=0,
                    end_world=dict(next_world),
                    end_bindings=frozenset(next_bindings),
                )
                report = run_task_preflight(
                    contract,
                    candidate,
                    max_depth=seed.max_depth,
                    terminal_profiles=terminal_profiles_by_id,
                    require_terminal_profile=True,
                    check_required_action_necessity=False,
                    binding_sources_by_id=binding_sources_by_id,
                    sat_result=bfs_sat,
                )
                if report.passed:
                    sampled.append(SampledTask(task=candidate, report=report))
                    seed_count += 1
                    if len(sampled) >= request.max_tasks:
                        return sampled
                    if seed_count >= per_seed_budget:
                        break

    return sampled


def cap_per_schema(
    sampled: list[SampledTask],
    request: SamplingRequestDoc,
    max_per_schema: int,
) -> list[SampledTask]:
    """Subsample tasks so no single schema dominates."""
    prefix_map: dict[str, str] = {}
    for schema in request.seed_schemas:
        prefix = schema.seed_id_template.split("{")[0]
        prefix_map[prefix] = schema.schema_id

    sorted_prefixes = sorted(prefix_map, key=len, reverse=True)

    schema_counts: dict[str, int] = {}
    result: list[SampledTask] = []
    for entry in sampled:
        seed_id = entry.task.task_id.rsplit("_d", 1)[0]
        schema_id = "unknown"
        for prefix in sorted_prefixes:
            if seed_id.startswith(prefix):
                schema_id = prefix_map[prefix]
                break
        count = schema_counts.get(schema_id, 0)
        if count < max_per_schema:
            result.append(entry)
            schema_counts[schema_id] = count + 1
    return result


def sampled_to_task_specs(sampled: list[SampledTask]) -> TaskSpecsDoc:
    """Convert sampled task wrappers to TaskSpecsDoc."""
    return TaskSpecsDoc(version=1, tasks=[entry.task for entry in sampled])
