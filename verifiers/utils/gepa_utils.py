"""Utility functions for GEPA optimization."""

import importlib.resources
import json
import logging
import math
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # type: ignore[unresolved-import]
except ImportError:
    import tomli as tomllib  # type: ignore[unresolved-import]

from openai import AsyncOpenAI, OpenAI

import verifiers as vf
from verifiers.adapters.gepa import GEPAAdapter
from verifiers.types import GEPAConfig
from verifiers.utils.eval_utils import save_rollout_results

logger = logging.getLogger(__name__)

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


def get_env_gepa_defaults(env_id: str) -> Dict[str, Any]:
    """Get GEPA config defaults from environment package's pyproject.toml.

    Returns dict with 'num_examples', 'num_val', and 'rollouts_per_example' keys if found,
    otherwise returns empty dict. All errors are silently handled.
    """
    defaults: Dict[str, Any] = {}
    module_name = env_id.replace("-", "_").split("/")[-1]

    try:
        # read pyproject.toml from installed package
        package_ref = importlib.resources.files(module_name)
        pyproject_file = package_ref / "pyproject.toml"

        if not pyproject_file.is_file():
            logger.debug(f"pyproject.toml not found in installed package {module_name}")
            return defaults

        with pyproject_file.open("rb") as f:
            pyproject_data = tomllib.load(f)

        # Extract [tool.verifiers.gepa] section
        gepa_config = (
            pyproject_data.get("tool", {}).get("verifiers", {}).get("gepa", {})
        )

        if "num_examples" in gepa_config:
            defaults["num_examples"] = gepa_config["num_examples"]
        if "num_val" in gepa_config:
            defaults["num_val"] = gepa_config["num_val"]
        if "rollouts_per_example" in gepa_config:
            defaults["rollouts_per_example"] = gepa_config["rollouts_per_example"]

        if defaults:
            logger.debug(
                f"Loaded GEPA defaults from {module_name} pyproject.toml: {defaults}"
            )
    except ModuleNotFoundError:
        logger.debug(f"Package {module_name} not installed")
    except Exception as e:
        logger.debug(
            f"Could not load GEPA defaults from {module_name} pyproject.toml: {e}"
        )

    return defaults


def ensure_env_dir_on_path(env_dir_path: str, env_id: str) -> None:
    """Add local environment directory to sys.path if present."""
    env_dir = Path(env_dir_path).resolve()
    if not env_dir.exists():
        return
    module_name = env_id.replace("-", "_").split("/")[-1]
    candidate = env_dir / module_name
    if candidate.exists():
        env_dir_str = str(env_dir)
        if env_dir_str not in sys.path:
            sys.path.insert(0, env_dir_str)
            logger.debug(f"Added {env_dir_str} to sys.path for environment loading")


async def save_candidate_rollouts(
    adapter: GEPAAdapter,
    candidate: dict[str, str],
    label: str,
    client: AsyncOpenAI,
    model: str,
    sampling_args: dict,
    num_examples: int,
    rollouts_per_example: int,
    max_concurrent: int,
    save_every: int,
    log_dir: Path,
) -> None:
    """
    Evaluate a candidate program and save rollout trajectories to disk.
    """
    if num_examples <= 0:
        logger.warning(
            "Skipping rollout saving for %s candidate because num_examples<=0", label
        )
        return

    env = adapter.build_program(candidate)
    rollouts_dir = log_dir / "rollouts" / label
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Saving %s candidate rollouts to %s (num_examples=%s, rollouts=%s)",
        label,
        rollouts_dir,
        num_examples,
        rollouts_per_example,
    )
    results = await env.evaluate(
        client=client,
        model=model,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        results_path=rollouts_dir,
        save_results=False,
        save_every=save_every,
    )
    save_rollout_results(results)


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


def print_optimization_results(result, log_dir: Path):
    """Print GEPA optimization results to console."""
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

    print("\n" + "=" * 80)
    print(f"Logs saved to: {log_dir}")
    print("=" * 80)


async def run_gepa_optimization(config: GEPAConfig):
    """
    Run GEPA optimization with provided configuration.

    Handles:
    - Adapter creation
    - Reflection client setup
    - GEPA optimize() call
    - Result saving and output

    Args:
        config: GEPAConfig with all optimization parameters

    Returns:
        GEPA optimization result
    """
    try:
        from gepa import optimize
    except ImportError:
        print("Error: GEPA is not installed.")
        print("Install with: uv add 'verifiers[gepa]'")
        sys.exit(1)

    from verifiers.utils.client_utils import setup_client

    # Setup task client
    client = setup_client(config.client_config)
    logger.debug("Initialized OpenAI client")

    # Load environment
    vf_env = vf.load_environment(env_id=config.env_id, **config.env_args)

    if isinstance(vf_env, vf.EnvGroup):
        raise ValueError(
            "GEPA optimization is not supported for EnvGroup environments. "
            "Optimize each environment individually, then combine them."
        )

    # Validate components
    for component in config.components_to_optimize:
        if component == "tool_descriptions":
            if not getattr(vf_env, "oai_tools", None):
                raise ValueError(
                    "Cannot optimize tool_descriptions: "
                    f"environment '{config.env_id}' has no tools configured."
                )
        elif not hasattr(vf_env, component):
            raise ValueError(
                f"Environment '{config.env_id}' is missing component '{component}'. "
                "Provide a component that exists on the environment."
            )

    # Create adapter
    adapter = GEPAAdapter(
        env=vf_env,
        client=client,
        model=config.model,
        sampling_args=config.sampling_args,
        components_to_optimize=config.components_to_optimize,
        num_rollouts_per_example=config.rollouts_per_example,
        max_concurrent=config.max_concurrent,
    )

    # Setup reflection client
    reflection_client_kwargs = {
        "api_key": config.reflection_api_key,
        "base_url": config.reflection_base_url,
    }
    if config.client_config.extra_headers:
        reflection_client_kwargs["default_headers"] = config.client_config.extra_headers
    reflection_client = OpenAI(**reflection_client_kwargs)
    logger.debug(
        "Reflection client configured for model %s at %s",
        config.reflection_model,
        config.reflection_base_url,
    )

    # Log initial component values
    logger.info("Initial component values:")
    for comp, value in config.seed_candidate.items():
        preview = value[:200] + "..." if len(value) > 200 else value
        logger.info(f"  {comp}: {preview}")

    # Run GEPA
    logger.info("=" * 80)
    logger.info("Starting GEPA optimization...")
    logger.info("=" * 80)

    try:
        result = optimize(
            seed_candidate=config.seed_candidate,
            trainset=config.trainset,
            valset=config.valset,
            adapter=adapter,
            max_metric_calls=config.max_metric_calls,
            reflection_lm=lambda x: call_reflection_model(
                reflection_client,
                x,
                config.reflection_model,
                config.reflection_temperature,
                config.reflection_max_tokens,
            ),
            reflection_minibatch_size=config.reflection_minibatch_size,
            run_dir=str(config.log_dir),
            track_best_outputs=config.track_stats,
            seed=config.seed,
            display_progress_bar=True,
        )
    except Exception as e:
        logger.error(f"GEPA optimization failed: {e}", exc_info=True)
        raise

    # Print results
    print_optimization_results(result, config.log_dir)

    # Prepare run configuration for saving
    run_config = {
        "env_id": config.env_id,
        "model": config.model,
        "reflection_model": config.reflection_model,
        "reflection_temperature": config.reflection_temperature,
        "components": config.components_to_optimize,
        "trainset_size": len(config.trainset),
        "valset_size": len(config.valset),
        "rollouts_per_example": config.rollouts_per_example,
        "max_metric_calls": config.max_metric_calls,
        "reflection_minibatch_size": config.reflection_minibatch_size,
        "seed": config.seed,
        "max_concurrent": config.max_concurrent,
    }

    # Save results
    save_optimized_components(
        config.env_id, result.best_candidate, config.seed_candidate, config.log_dir
    )
    save_optimization_metrics(config.env_id, result, config.log_dir, run_config)

    # Save rollouts if requested
    if config.save_results:
        save_every = config.save_every if config.save_every > 0 else -1
        val_examples_for_logging = (
            config.num_val if config.num_val > 0 else config.num_examples
        )

        async def save_all_candidates():
            await save_candidate_rollouts(
                adapter=adapter,
                candidate=config.seed_candidate,
                label="seed",
                client=client,
                model=config.model,
                sampling_args=config.sampling_args,
                num_examples=val_examples_for_logging,
                rollouts_per_example=config.rollouts_per_example,
                max_concurrent=config.max_concurrent,
                save_every=save_every,
                log_dir=config.log_dir,
            )
            await save_candidate_rollouts(
                adapter=adapter,
                candidate=result.best_candidate,
                label="best",
                client=client,
                model=config.model,
                sampling_args=config.sampling_args,
                num_examples=val_examples_for_logging,
                rollouts_per_example=config.rollouts_per_example,
                max_concurrent=config.max_concurrent,
                save_every=save_every,
                log_dir=config.log_dir,
            )

        try:
            await save_all_candidates()
        except RuntimeError as exc:
            logger.error(f"Failed to save rollout trajectories: {exc}")

    logger.info("GEPA optimization completed successfully!")
    return result
