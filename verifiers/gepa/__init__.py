"""
GEPA (Genetic-Pareto) integration for Verifiers.

This module provides adapter, utilities, and templates for optimizing
Verifiers environments using the GEPA reflection-based optimization algorithm.

Main components:
- GEPAAdapter: Bridges Verifiers environments with GEPA optimization
- run_gepa_optimization: High-level function to run GEPA on an environment
- TOOL_DESCRIPTION_PROMPT_TEMPLATE: Template for tool description optimization
"""

from .adapter import GEPAAdapter
from .templates import TOOL_DESCRIPTION_PROMPT_TEMPLATE
from .utils import (
    auto_budget_to_metric_calls,
    call_reflection_model,
    ensure_env_dir_on_path,
    get_env_gepa_defaults,
    prepare_gepa_dataset,
    print_optimization_results,
    run_gepa_optimization,
    save_candidate_rollouts,
    save_optimized_components,
    save_optimization_metrics,
)

__all__ = [
    # Core adapter
    "GEPAAdapter",
    # Templates
    "TOOL_DESCRIPTION_PROMPT_TEMPLATE",
    # Main optimization function
    "run_gepa_optimization",
    # Utility functions
    "auto_budget_to_metric_calls",
    "call_reflection_model",
    "ensure_env_dir_on_path",
    "get_env_gepa_defaults",
    "prepare_gepa_dataset",
    "print_optimization_results",
    "save_candidate_rollouts",
    "save_optimized_components",
    "save_optimization_metrics",
]
