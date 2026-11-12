"""Offline submission regrading against an explicitly selected taskset."""

from pathlib import Path
from uuid import uuid4

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.cli.run import RunConfig
from verifiers.v1.configs.cli.validate import CheckTimeoutConfig
from verifiers.v1.configs.runtime import PrimeConfig, RuntimeConfig
from verifiers.v1.configs.select import SelectCLIConfig
from verifiers.v1.configs.taskset import TasksetConfig


class RegradeConfig(BaseConfig):
    source: Path
    """Completed run directory containing traces.jsonl and verified artifact archives."""
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    """Current task definitions; submissions match by unique task id, then name.
    Task data must record a stable solver_digest covering its solver inputs;
    Harbor supplies this. Missing provenance requires a new solver run."""
    select: SelectCLIConfig = SelectCLIConfig()
    runtime: RuntimeConfig = PrimeConfig()
    """Provider policy for fresh verifier environments; no solver is provisioned."""
    timeout: CheckTimeoutConfig = CheckTimeoutConfig()
    max_concurrent: int = Field(
        16, ge=1, validation_alias=AliasChoices("max_concurrent", "c")
    )
    run: RunConfig = Field(default_factory=RunConfig)
    output_dir: Path = Field(
        Path("outputs"), validation_alias=AliasChoices("output_dir", "o")
    )
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))

    @model_validator(mode="before")
    @classmethod
    def resolve_taskset(cls, data):
        from verifiers.v1.utils.loaders import narrow_plugin_field, taskset_config_type

        narrow_plugin_field(data, "taskset", taskset_config_type)
        return data

    @model_validator(mode="after")
    def name_run(self):
        if self.run.name is None:
            self.run.name = f"{self.taskset.name}--regrade--{uuid4().hex[:8]}".lower()
        if self.run.dir is None:
            self.run.dir = self.run.name
        return self
