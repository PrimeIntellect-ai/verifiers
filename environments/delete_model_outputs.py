#!/usr/bin/env python3
"""
Delete model outputs from an environment's outputs directory.

Usage:
    python delete_model_outputs.py --env oolong --list
    python delete_model_outputs.py --env oolong --model gpt-5-mini
    python delete_model_outputs.py --env oolong --model gpt-5-mini --dry-run
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


def list_models(env_name: str):
    """List all models with outputs for an environment."""
    outputs_dir = get_outputs_dir(env_name)
    if outputs_dir is None:
        return

    evals_dir = outputs_dir / "evals"
    if not evals_dir.exists():
        print(f"No evals directory found in {outputs_dir}")
        return

    # Collect model info: model_name -> (dir_path, run_count)
    models: dict[str, tuple[Path, int]] = {}

    for model_dir in evals_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Find model name from metadata
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            metadata_file = run_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                model_name = metadata.get("model", "unknown")
                run_count = len([r for r in model_dir.iterdir() if r.is_dir()])
                models[model_name] = (model_dir, run_count)
                break

    if not models:
        print(f"No model outputs found for environment '{env_name}'")
        return

    print(f"Models with outputs in '{env_name}':")
    for model_name, (dir_path, run_count) in sorted(models.items()):
        print(f"  {model_name} ({run_count} run(s))")


def find_model_dirs(outputs_dir: Path, env_name: str, model_name: str) -> list[Path]:
    """Find all output directories for a given model."""
    evals_dir = outputs_dir / "evals"
    if not evals_dir.exists():
        return []

    matching_dirs = []

    # Pattern: <env_name>--<model_name>
    # Model names can contain slashes, so we check metadata.json for exact match
    for model_dir in evals_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Check if directory name matches pattern
        expected_prefix = f"{env_name}--"
        if not model_dir.name.startswith(expected_prefix):
            continue

        # Verify by checking metadata in subdirectories
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
            metadata_file = run_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                if metadata.get("model") == model_name:
                    matching_dirs.append(model_dir)
                    break  # Found a match, no need to check more runs

    return matching_dirs


def delete_model_outputs(
    env_name: str, model_name: str, dry_run: bool = False, force: bool = False
):
    """Delete all outputs for a model in an environment."""
    outputs_dir = get_outputs_dir(env_name)
    if outputs_dir is None:
        return

    # Find matching directories
    matching_dirs = find_model_dirs(outputs_dir, env_name, model_name)

    if not matching_dirs:
        print(f"No outputs found for model '{model_name}' in environment '{env_name}'")
        return

    print(f"Found {len(matching_dirs)} output directory(s) for model '{model_name}':")
    for d in matching_dirs:
        # Count runs
        runs = [r for r in d.iterdir() if r.is_dir()]
        print(f"  {d} ({len(runs)} run(s))")

    if dry_run:
        print("\n[DRY RUN] No files deleted.")
        return

    # Confirm deletion unless force is set
    if not force:
        response = input("\nDelete these directories? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    # Delete
    for d in matching_dirs:
        print(f"Deleting {d}...")
        shutil.rmtree(d)

    print(f"Deleted {len(matching_dirs)} directory(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Delete model outputs from an environment"
    )
    parser.add_argument(
        "--env", "-e", required=True, help="Environment name (e.g., oolong)"
    )
    parser.add_argument("--model", "-m", help="Model name (e.g., gpt-5-mini)")
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
        list_models(args.env)
    elif args.model:
        delete_model_outputs(args.env, args.model, args.dry_run, args.force)
    else:
        parser.error("Either --list or --model is required")


if __name__ == "__main__":
    main()
