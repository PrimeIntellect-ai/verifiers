#!/usr/bin/env python3
"""
GEPA optimization script for Verifiers environments.

Usage:
    vf-gepa wordle --auto light
    vf-gepa wiki-search --auto heavy --components system_prompt tool_descriptions
    vf-gepa my-env --max-metric-calls 1000 -n 100 --num-val 30
"""

import argparse
import json
import logging
import math
import os
import sys
import textwrap
import uuid
from pathlib import Path

try:
    from gepa import optimize
except ImportError:
    print("Error: GEPA is not installed.")
    print("Install with: uv add 'verifiers[gepa]'")
    sys.exit(1)


from openai import OpenAI

import verifiers as vf
from verifiers.adapters.gepa import GEPAAdapter
from verifiers.types import ClientConfig
from verifiers.utils.client_utils import setup_client

logger = logging.getLogger("gepa")

# Auto-budget constants for clarity and tuning
AUTO_BUDGET_CANDIDATES = {
    "light": 6,
    "medium": 12,
    "heavy": 18,
}
TRIAL_LOG_BASE_MULTIPLIER = 2.0
TRIAL_COMPONENT_MULTIPLIER = 2
TRIAL_LINEAR_MULTIPLIER = 1.5
BOOTSTRAP_TRIALS_PER_CANDIDATE = 5


def auto_budget_to_metric_calls(
    auto: str,
    num_components: int,
    valset_size: int,
    minibatch_size: int = 3,
    full_eval_steps: int = 5,
) -> int:
    """
    Convert auto budget (light/medium/heavy) to max_metric_calls.

    This replicates GEPA's auto_budget calculation for consistency.

    Args:
        auto: Budget level ('light', 'medium', or 'heavy')
        num_components: Number of components being optimized
        valset_size: Size of validation set
        minibatch_size: Reflection minibatch size
        full_eval_steps: Steps between full validations

    Returns:
        Maximum number of metric calls
    """
    num_candidates = AUTO_BUDGET_CANDIDATES[auto]

    # Calculate number of trials using log-growth vs. linear fallback
    log_trials = (
        TRIAL_LOG_BASE_MULTIPLIER
        * (num_components * TRIAL_COMPONENT_MULTIPLIER)
        * math.log2(num_candidates)
    )
    linear_trials = TRIAL_LINEAR_MULTIPLIER * num_candidates
    num_trials = int(max(log_trials, linear_trials))

    V = valset_size
    N = num_trials
    M = minibatch_size
    m = full_eval_steps

    # Initial full evaluation on the default program
    total = V

    # Assume a handful of bootstrap trials per candidate
    total += num_candidates * BOOTSTRAP_TRIALS_PER_CANDIDATE

    # N minibatch evaluations
    total += N * M

    if N == 0:
        return total

    # Periodic full evals
    periodic_fulls = (N + 1) // m + 1
    extra_final = 1 if N < m else 0

    total += (periodic_fulls + extra_final) * V

    logger.info(
        f"Auto budget '{auto}' → ~{num_candidates} candidates, "
        f"~{total} metric calls (~{total // (V or 1)} full evals)"
    )

    return total


def prepare_gepa_dataset(dataset) -> list[dict]:
    """
    Convert HuggingFace Dataset to GEPA format.

    GEPA expects a list of dicts with keys like 'question', 'answer', 'info', 'task'.
    """
    if dataset is None:
        return []

    examples = []
    for item in dataset:
        example = {
            "question": item.get("question", item.get("prompt", "")),
            "answer": item.get("answer", ""),
            "task": item.get("task", "default"),
            "info": item.get("info", {}),
        }
        examples.append(example)

    return examples


def call_reflection_model(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float = 1.0,
    max_tokens: int | None = None,
) -> str:
    """
    Call reflection model to generate proposal.

    This is a wrapper around the API call for GEPA's reflection phase.
    """
    try:
        request_args = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens is not None:
            request_args["max_tokens"] = max_tokens
        response = client.chat.completions.create(**request_args)
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Error calling reflection model: {e}")
        raise


def save_optimized_components(
    env_id: str,
    best_candidate: dict[str, str],
    seed_candidate: dict[str, str],
    output_dir: Path,
):
    """Save optimized components to disk for future use."""
    output_file = output_dir / f"{env_id}_optimized.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(best_candidate, f, indent=2)

    logger.info(f"Saved optimized components to: {output_file}")

    # Also save the original (seed) components for comparison
    original_file = output_dir / f"{env_id}_original.json"
    with open(original_file, "w") as f:
        json.dump(seed_candidate, f, indent=2)

    logger.info(f"Saved original components to: {original_file}")


def save_optimization_metrics(
    env_id: str,
    result,
    output_dir: Path,
    run_config: dict,
):
    """Save optimization metrics and configuration for analysis."""
    from datetime import datetime

    metrics_file = output_dir / f"{env_id}_metrics.json"

    metrics = {
        # Run configuration
        "config": run_config,
        # Timestamps
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        # Results
        "val_aggregate_scores": result.val_aggregate_scores,
        "num_candidates": len(result.candidates),
        "best_val_score": (
            float(max(result.val_aggregate_scores))
            if result.val_aggregate_scores
            else 0.0
        ),
        "initial_val_score": (
            float(result.val_aggregate_scores[0])
            if result.val_aggregate_scores
            else 0.0
        ),
        "improvement": (
            float(max(result.val_aggregate_scores) - result.val_aggregate_scores[0])
            if len(result.val_aggregate_scores) > 0
            else 0.0
        ),
        "candidates_history": [
            {
                "iteration": i,
                "score": float(score),
            }
            for i, score in enumerate(result.val_aggregate_scores)
        ],
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved optimization metrics to: {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run GEPA prompt optimization on Verifiers environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Light optimization (quick test)
  vf-gepa wordle --auto light

  # Heavy optimization with tool descriptions
  vf-gepa wiki-search --auto heavy --components system_prompt tool_descriptions

  # Custom configuration
  vf-gepa my-env --max-metric-calls 1000 -n 100 --num-val 30
        """,
    )

    # Environment args
    parser.add_argument(
        "env_id", type=str, help="Environment ID (e.g., wordle, wiki-search)"
    )
    parser.add_argument(
        "--env-args",
        "-a",
        default="{}",
        help="JSON dict of keyword args forwarded to vf.load_environment",
    )

    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=50,
        help="Number of training examples (default: 50)",
    )

    parser.add_argument(
        "--num-val",
        type=int,
        default=20,
        help="Number of validation examples (default: 20)",
    )

    # GEPA budget (mutually exclusive)
    budget_group = parser.add_mutually_exclusive_group(required=True)
    budget_group.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        help="Auto budget: light (~6 candidates), medium (~12), heavy (~18)",
    )
    budget_group.add_argument(
        "--max-metric-calls", type=int, help="Maximum total metric calls budget"
    )

    # GEPA configuration
    parser.add_argument(
        "--reflection-model",
        default="gpt-4o",
        help="Model for reflection/proposal (default: gpt-4o)",
    )

    parser.add_argument(
        "--reflection-temperature",
        type=float,
        default=1.0,
        help="Temperature for reflection model (default: 1.0)",
    )

    parser.add_argument(
        "--reflection-base-url",
        default=None,
        help="Base URL for reflection model API (default: task client base URL)",
    )

    parser.add_argument(
        "--reflection-api-key-var",
        default="OPENAI_API_KEY",
        help="Env var that stores the reflection API key (default: OPENAI_API_KEY)",
    )

    parser.add_argument(
        "--reflection-max-tokens",
        type=int,
        default=8000,
        help="Max tokens for reflection completions (default: 8000)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="gpt-4o-mini",
        help="Model to optimize (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        default="OPENAI_API_KEY",
        help="Environment variable containing the task model API key",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        default="https://api.openai.com/v1",
        help="Base URL for the task model API (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--header",
        action="append",
        dest="headers",
        default=None,
        help="Additional HTTP header for the task model client. Format: 'Name: Value'. Repeatable.",
    )

    parser.add_argument(
        "--components",
        nargs="+",
        default=["system_prompt"],
        help="Components to optimize (default: system_prompt)",
    )

    parser.add_argument(
        "--reflection-minibatch-size",
        type=int,
        default=3,
        help="Number of examples per reflection step (default: 3)",
    )

    parser.add_argument(
        "--rollouts-per-example",
        type=int,
        default=1,
        help="Number of rollouts per example (default: 1)",
    )

    # Model configuration
    parser.add_argument(
        "-T",
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for task model (default: 1.0)",
    )

    parser.add_argument(
        "-t",
        "--max-tokens",
        type=int,
        default=8096,
        help="Max tokens for task model (default: 8096)",
    )

    # Logging
    parser.add_argument(
        "--log-dir",
        help="Directory for GEPA logs (default: ./gepa_results/<env_id>/<run_id>)",
    )

    parser.add_argument(
        "--track-stats",
        action="store_true",
        help="Track detailed optimization statistics",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    try:
        env_args = json.loads(args.env_args)
        if not isinstance(env_args, dict):
            raise TypeError("env args must be a JSON object")
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            "--env-args must be valid JSON representing a dictionary"
        ) from exc

    task_client_headers: dict[str, str] | None = None
    if args.headers:
        task_client_headers = {}
        for header in args.headers:
            if ":" not in header:
                raise ValueError(
                    "Headers must be provided in the format 'Name: Value'."
                )
            key, value = header.split(":", 1)
            task_client_headers[key.strip()] = value.strip()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Silence noisy third-party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info(f"Starting GEPA optimization for environment: {args.env_id}")
    logger.info(f"Components to optimize: {args.components}")

    # Setup client
    client_config_kwargs = {
        "api_key_var": args.api_key_var,
        "api_base_url": args.api_base_url,
    }
    if task_client_headers is not None:
        client_config_kwargs["extra_headers"] = task_client_headers

    client_config = ClientConfig(**client_config_kwargs)
    client = setup_client(client_config)
    logger.debug("Initialized OpenAI client")

    # Load environment
    vf_env = vf.load_environment(env_id=args.env_id, **env_args)

    if isinstance(vf_env, vf.EnvGroup):
        raise ValueError(
            "GEPA optimization is not supported for EnvGroup environments. "
            "Optimize each environment individually, then combine them."
        )

    for component in args.components:
        if component == "tool_descriptions":
            if not getattr(vf_env, "oai_tools", None):
                raise ValueError(
                    "Cannot optimize tool_descriptions: "
                    f"environment '{args.env_id}' has no tools configured."
                )
        elif not hasattr(vf_env, component):
            raise ValueError(
                f"Environment '{args.env_id}' is missing component '{component}'. "
                "Provide a component that exists on the environment."
            )

    # Setup sampling args
    sampling_args = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # Create adapter
    adapter = GEPAAdapter(
        env=vf_env,
        client=client,
        model=args.model,
        sampling_args=sampling_args,
        components_to_optimize=args.components,
        num_rollouts_per_example=args.rollouts_per_example,
        max_concurrent=32,
    )

    # Prepare datasets
    logger.info(f"Loading {args.num_examples} training examples")
    logger.info(f"Loading {args.num_val} validation examples")
    if vf_env.eval_dataset is not None:
        train_dataset_raw = vf_env.get_dataset(n=args.num_examples, seed=args.seed)
        val_dataset_raw = vf_env.get_eval_dataset(n=args.num_val, seed=args.seed + 1)
    else:
        total_requested = max(args.num_examples, 0) + max(args.num_val, 0)
        base_dataset = vf_env.get_dataset(n=total_requested, seed=args.seed)
        base_examples = (
            base_dataset.to_list()
            if hasattr(base_dataset, "to_list")
            else list(base_dataset)
        )
        train_dataset_raw = (
            base_examples[: args.num_examples]
            if args.num_examples > 0
            else base_examples
        )
        val_dataset_raw = (
            base_examples[args.num_examples : args.num_examples + args.num_val]
            if args.num_val > 0
            else []
        )
        logger.debug(
            "Eval dataset missing; derived %s validation examples from train split",
            len(val_dataset_raw),
        )

    trainset = prepare_gepa_dataset(train_dataset_raw)
    valset = prepare_gepa_dataset(val_dataset_raw)

    if args.num_examples > 0 and not trainset:
        raise ValueError(
            "Training dataset is empty - check environment configuration and filters"
        )
    if args.num_val > 0 and not valset:
        raise ValueError(
            "Validation dataset is empty - check environment configuration and filters"
        )

    logger.info(f"Training set: {len(trainset)} examples")
    logger.info(f"Validation set: {len(valset)} examples")

    reflection_api_key_var = args.reflection_api_key_var or client_config.api_key_var
    reflection_api_key = os.getenv(reflection_api_key_var)
    if not reflection_api_key:
        raise ValueError(
            f"{reflection_api_key_var} environment variable not set for reflection client"
        )
    reflection_base_url = args.reflection_base_url
    if not reflection_base_url:
        base_url = getattr(client, "base_url", None)
        reflection_base_url = str(base_url) if base_url else "https://api.openai.com/v1"

    reflection_client_kwargs = {
        "api_key": reflection_api_key,
        "base_url": reflection_base_url,
    }
    if task_client_headers:
        reflection_client_kwargs["default_headers"] = task_client_headers
    reflection_client = OpenAI(**reflection_client_kwargs)
    logger.debug(
        "Reflection client configured for model %s at %s",
        args.reflection_model,
        reflection_base_url,
    )

    # Extract seed candidate (initial component values)
    seed_candidate = {}
    for comp in args.components:
        if comp == "tool_descriptions":
            # Extract tool descriptions
            if hasattr(vf_env, "oai_tools") and vf_env.oai_tools:
                for i, tool in enumerate(vf_env.oai_tools):
                    seed_candidate[f"tool_{i}_description"] = tool["function"][
                        "description"
                    ]
        elif hasattr(vf_env, comp):
            seed_candidate[comp] = getattr(vf_env, comp)
        else:
            logger.warning(f"Environment doesn't have component '{comp}', skipping")

    if not seed_candidate:
        logger.error("No valid components found to optimize!")
        return

    logger.info("Initial component values:")
    for comp, value in seed_candidate.items():
        preview = value[:200] + "..." if len(value) > 200 else value
        logger.info(f"  {comp}: {preview}")

    # Setup log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        run_id = str(uuid.uuid4())[:8]
        log_dir = Path(f"./gepa_results/{args.env_id}/{run_id}")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Log directory: {log_dir}")

    # Convert auto budget to max_metric_calls if needed
    if args.auto:
        max_metric_calls = auto_budget_to_metric_calls(
            auto=args.auto,
            num_components=len(seed_candidate),
            valset_size=len(valset),
            minibatch_size=args.reflection_minibatch_size,
        )
    else:
        max_metric_calls = args.max_metric_calls

    logger.info(f"Budget: {max_metric_calls} metric calls total")

    # Run GEPA
    logger.info("=" * 80)
    logger.info("Starting GEPA optimization...")
    logger.info("=" * 80)

    try:
        result = optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            max_metric_calls=max_metric_calls,
            reflection_lm=lambda x: call_reflection_model(
                reflection_client,
                x,
                args.reflection_model,
                args.reflection_temperature,
                args.reflection_max_tokens,
            ),
            reflection_minibatch_size=args.reflection_minibatch_size,
            run_dir=str(log_dir),
            track_best_outputs=args.track_stats,
            seed=args.seed,
            display_progress_bar=True,
        )
    except Exception as e:
        logger.error(f"GEPA optimization failed: {e}", exc_info=True)
        raise

    # Print results
    print("\n" + "=" * 80)
    print("GEPA OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best validation score: {max(result.val_aggregate_scores):.3f}")
    print(f"Initial validation score: {result.val_aggregate_scores[0]:.3f}")
    print(
        f"Improvement: {max(result.val_aggregate_scores) - result.val_aggregate_scores[0]:.3f}"
    )
    print(f"Total candidates explored: {len(result.candidates)}")
    print("\nOptimized components:")
    print("-" * 80)

    for comp, text in result.best_candidate.items():
        print(f"\n{comp}:")
        print(textwrap.indent(text, "  "))

    # Prepare run configuration for saving
    run_config = {
        "env_id": args.env_id,
        "model": args.model,
        "reflection_model": args.reflection_model,
        "reflection_temperature": args.reflection_temperature,
        "components": args.components,
        "trainset_size": len(trainset),
        "valset_size": len(valset),
        "rollouts_per_example": args.rollouts_per_example,
        "max_metric_calls": max_metric_calls,
        "reflection_minibatch_size": args.reflection_minibatch_size,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    # Save results
    save_optimized_components(
        args.env_id, result.best_candidate, seed_candidate, log_dir
    )
    save_optimization_metrics(args.env_id, result, log_dir, run_config)

    print("\n" + "=" * 80)
    print(f"Logs saved to: {log_dir}")
    print("=" * 80)

    logger.info("GEPA optimization completed successfully!")


if __name__ == "__main__":
    main()
