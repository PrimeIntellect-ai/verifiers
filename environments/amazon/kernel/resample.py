#!/usr/bin/env python3
"""Re-run BFS sampling from graph_contract.yaml + sampling_request.yaml.

Outputs task_specs.sampled.yaml in the amazon environment root.

Usage (from environments/amazon/):
    python -m kernel.resample
    python -m kernel.resample --max-per-schema 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .loaders import load_graph_contract, load_sampling_request
from .sampler import cap_per_schema, sample_task_intents, sampled_to_task_specs

_KERNEL_DIR = Path(__file__).parent
_ENV_DIR = _KERNEL_DIR.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-sample BFS task intents")
    parser.add_argument(
        "--max-per-schema",
        type=int,
        default=None,
        help="Cap tasks per seed schema (default: no cap)",
    )
    args = parser.parse_args()

    contract_path = _KERNEL_DIR / "graph_contract.yaml"
    request_path = _KERNEL_DIR / "sampling_request.yaml"
    out_path = _ENV_DIR / "task_specs.sampled.yaml"

    contract = load_graph_contract(contract_path)
    request = load_sampling_request(request_path)

    print(
        f"Contract: {len(contract.actions)} actions, {len(contract.projection_fields)} projection fields"
    )
    print(
        f"Request: {len(request.seeds)} seeds, {len(request.terminal_profiles)} terminal profiles"
    )

    sampled = sample_task_intents(contract, request)
    print(f"BFS produced {len(sampled)} passing tasks")

    if args.max_per_schema:
        sampled = cap_per_schema(sampled, request, args.max_per_schema)
        print(f"After cap_per_schema({args.max_per_schema}): {len(sampled)} tasks")

    # Print summary by terminal profile
    by_profile: dict[str, int] = {}
    for st in sampled:
        pid = st.task.terminal_profile_id or "none"
        by_profile[pid] = by_profile.get(pid, 0) + 1
    for pid, count in sorted(by_profile.items()):
        print(f"  {pid}: {count} tasks")

    specs = sampled_to_task_specs(sampled)
    out_path.write_text(
        yaml.safe_dump(specs.model_dump(mode="python"), sort_keys=False)
    )
    print(f"\nWrote {len(sampled)} tasks to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
