#!/usr/bin/env python3
"""
Delete model outputs from an environment's outputs directory.

Usage:
    python delete_model_outputs.py -e oolong --list
    python delete_model_outputs.py -e oolong -m gpt-5-mini
    python delete_model_outputs.py -e oolong -m gpt-5-mini -M rlm
    python delete_model_outputs.py -e oolong -m gpt-5-mini --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def get_outputs_dir(env_name: str) -> Path | None:
    """Get the outputs directory for an environment."""
    script_dir = Path(__file__).parent
    env_dir = script_dir / env_name

    if not env_dir.exists():
        print(f"Error: Environment directory not found: {env_dir}")
        return None

    outputs_dir = env_dir / "outputs"
    if not outputs_dir.exists():
        print(f"No outputs directory found in {env_dir}")
        return None

    return outputs_dir


def get_mode_from_metadata(metadata: dict) -> str:
    """Extract mode from metadata env_args."""
    env_args = metadata.get("env_args", {})
    use_rlm = env_args.get("use_rlm", False)
    include_tips = env_args.get("include_env_tips", False)

    if not use_rlm:
        return "standard"
    elif include_tips:
        return "rlm_tips"
    else:
        return "rlm"


def list_models(env_name: str):
    """List all models with outputs for an environment."""
    outputs_dir = get_outputs_dir(env_name)
    if outputs_dir is None:
        return

    evals_dir = outputs_dir / "evals"
    if not evals_dir.exists():
        print(f"No evals directory found in {outputs_dir}")
        return

    # Collect model info: model_name -> {mode -> count}
    models: dict[str, dict[str, int]] = {}

    for model_dir in evals_dir.iterdir():
        if not model_dir.is_dir():
            continue

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            metadata_file = run_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                model_name = metadata.get("model", "unknown")
                mode = get_mode_from_metadata(metadata)

                if model_name not in models:
                    models[model_name] = {}
                models[model_name][mode] = models[model_name].get(mode, 0) + 1

    if not models:
        print(f"No model outputs found for environment '{env_name}'")
        return

    print(f"Models with outputs in '{env_name}':")
    for model_name in sorted(models.keys()):
        mode_counts = models[model_name]
        total = sum(mode_counts.values())
        mode_str = ", ".join(f"{m}:{c}" for m, c in sorted(mode_counts.items()))
        print(f"  {model_name} ({total} run(s): {mode_str})")


def find_model_runs(
    outputs_dir: Path, env_name: str, model_name: str, mode: str | None = None
) -> list[tuple[Path, list[Path]]]:
    """Find all output directories for a given model, optionally filtered by mode.

    Returns list of (model_dir, [matching_run_dirs]) tuples.
    """
    evals_dir = outputs_dir / "evals"
    if not evals_dir.exists():
        return []

    results = []

    for model_dir in evals_dir.iterdir():
        if not model_dir.is_dir():
            continue

        expected_prefix = f"{env_name}--"
        if not model_dir.name.startswith(expected_prefix):
            continue

        matching_runs = []
        model_matched = False

        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            metadata_file = run_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                if metadata.get("model") == model_name:
                    model_matched = True
                    # If mode filter specified, check it
                    if mode is None:
                        matching_runs.append(run_dir)
                    elif get_mode_from_metadata(metadata) == mode:
                        matching_runs.append(run_dir)

        if model_matched and matching_runs:
            results.append((model_dir, matching_runs))

    return results


def delete_model_outputs(
    env_name: str,
    model_name: str,
    mode: str | None = None,
    dry_run: bool = False,
    force: bool = False,
):
    """Delete all outputs for a model (and optionally mode) in an environment."""
    outputs_dir = get_outputs_dir(env_name)
    if outputs_dir is None:
        return

    results = find_model_runs(outputs_dir, env_name, model_name, mode)

    if not results:
        mode_str = f" in mode '{mode}'" if mode else ""
        print(
            f"No outputs found for model '{model_name}'{mode_str} in environment '{env_name}'"
        )
        return

    total_runs = sum(len(runs) for _, runs in results)
    mode_str = f" (mode: {mode})" if mode else ""
    print(f"Found {total_runs} run(s) for model '{model_name}'{mode_str}:")

    for _, run_dirs in results:
        for run_dir in run_dirs:
            metadata_file = run_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            run_mode = get_mode_from_metadata(metadata)
            subset = metadata.get("env_args", {}).get("subset", "?")
            print(f"  {run_dir.name} ({run_mode}, {subset})")

    if dry_run:
        print("\n[DRY RUN] No files deleted.")
        return

    if not force:
        response = input("\nDelete these runs? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Delete only the matching runs (not entire model dir if mode filtered)
    deleted = 0
    for model_dir, run_dirs in results:
        for run_dir in run_dirs:
            print(f"Deleting {run_dir}...")
            shutil.rmtree(run_dir)
            deleted += 1

        # Clean up empty model dir
        remaining = [r for r in model_dir.iterdir() if r.is_dir()]
        if not remaining:
            print(f"Removing empty directory {model_dir}...")
            model_dir.rmdir()

    print(f"Deleted {deleted} run(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Delete model outputs from an environment"
    )
    parser.add_argument(
        "--environment", "-e", required=True, help="Environment name (e.g., oolong)"
    )
    parser.add_argument("--model", "-m", help="Model name (e.g., gpt-5-mini)")
    parser.add_argument(
        "--mode",
        "-M",
        choices=["standard", "rlm", "rlm_tips"],
        help="Filter by mode (standard, rlm, or rlm_tips)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all models with outputs for the environment",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="Skip confirmation prompt"
    )
    args = parser.parse_args()

    if args.list:
        list_models(args.environment)
    elif args.model:
        delete_model_outputs(
            args.environment, args.model, args.mode, args.dry_run, args.force
        )
    else:
        parser.error("Either --list or --model is required")


if __name__ == "__main__":
    main()
