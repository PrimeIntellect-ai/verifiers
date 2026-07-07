from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from verifiers.v1.types import EnvId


class GEPAEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: EnvId
    args: dict[str, Any] = Field(default_factory=dict)
    extra_env_kwargs: dict[str, Any] = Field(default_factory=dict)


class GEPAOptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_calls: int = Field(
        default=500, ge=1, validation_alias=AliasChoices("max_calls", "B")
    )
    minibatch_size: int = Field(default=3, ge=1)
    perfect_score: float | None = None
    state_columns: list[str] = Field(default_factory=list)
    num_train: int = Field(
        default=100, ge=-1, validation_alias=AliasChoices("num_train", "n")
    )
    num_val: int = Field(
        default=50, ge=-1, validation_alias=AliasChoices("num_val", "N")
    )
    max_concurrent: int = Field(default=32, ge=1)
    seed: int = 0


class GEPAConfig(BaseModel):
    """Typed contract shared by GEPA's Python and CLI surfaces."""

    model_config = ConfigDict(extra="forbid")

    id: EnvId | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    extra_env_kwargs: dict[str, Any] = Field(default_factory=dict)
    env: list[GEPAEnvConfig] = Field(default_factory=list)
    model: str = Field(
        default="openai/gpt-4.1-mini", validation_alias=AliasChoices("model", "m")
    )
    reflection_model: str | None = Field(
        default=None, validation_alias=AliasChoices("reflection_model", "M")
    )
    endpoints_path: Path = Path("./configs/endpoints.toml")
    api_key_var: str | None = None
    api_base_url: str | None = None
    env_dir_path: Path = Path("./environments")
    gepa: GEPAOptimizationConfig = Field(default_factory=GEPAOptimizationConfig)
    sampling: dict[str, Any] = Field(default_factory=dict)
    verbose: bool = False
    run_dir: Path | None = None
    save_results: bool = True
    tui: bool = False

    @model_validator(mode="before")
    @classmethod
    def wrap_single_env(cls, data):
        """Accept a single `[env]` table by wrapping it into a one-element `[[env]]` list."""
        if isinstance(data, dict) and isinstance(data.get("env"), dict):
            data["env"] = [data["env"]]
        return data

    @model_validator(mode="after")
    def select_environments(self) -> "GEPAConfig":
        if bool(self.id) == bool(self.env):
            raise ValueError("set exactly one of id or [[env]]")
        return self

    @property
    def environments(self) -> list[GEPAEnvConfig]:
        if self.id:
            return [
                GEPAEnvConfig(
                    id=self.id,
                    args=self.args,
                    extra_env_kwargs=self.extra_env_kwargs,
                )
            ]
        return self.env

    @property
    def environment_label(self) -> str:
        if self.id:
            return self.id
        return "+".join(env.id for env in self.env)
