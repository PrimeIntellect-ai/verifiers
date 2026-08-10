from dataclasses import dataclass, field
from pathlib import Path

from verifiers.types import ClientConfig, SamplingArgs


@dataclass
class GEPAConfig:
    """Configuration for GEPA optimization."""

    # Environment
    env_id: str
    env_args: dict = field(default_factory=dict)

    # Models
    model: str = ""  # Model for rollouts
    reflection_model: str | None = None  # Model for reflection (defaults to model)
    client_config: ClientConfig = field(default_factory=ClientConfig)

    # Dataset sizes
    num_train_examples: int = 100
    num_val_examples: int = 50

    # GEPA optimization
    max_metric_calls: int = 500
    reflection_minibatch_size: int = 3
    initial_prompt: str | None = None  # None = use env.system_prompt

    # Reflective dataset
    state_columns: list[str] = field(default_factory=list)

    # Execution
    sampling_args: SamplingArgs = field(default_factory=dict)
    max_concurrent: int = 32

    # Output
    run_dir: Path | None = None
    seed: int = 0
    verbose: bool = False

    # Saving
    save_results: bool = True  # Save final results to disk
