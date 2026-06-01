import json
from typing import Literal, TypeAlias, cast

from pydantic import ConfigDict, StrictBool, model_validator
from typing_extensions import Self

from .config import Config, validate_serializable_value
from .types import ConfigData, JsonValue

ArtifactFormat: TypeAlias = Literal["text", "json"]


class ArtifactConfig(Config):
    path: str
    format: ArtifactFormat = "text"
    key: str | None = None
    optional: StrictBool = False

    def data(self) -> ConfigData:
        data = self.model_dump(exclude_none=True)
        if self.optional is False:
            data.pop("optional", None)
        return cast(ConfigData, data)

    def parse(self, content: str) -> JsonValue:
        if self.format == "json":
            value = cast(JsonValue, json.loads(content))
        elif self.format == "text":
            value = content
        else:
            raise AssertionError(f"Unsupported artifact format: {self.format!r}")
        if self.key is None:
            return value
        if not isinstance(value, dict):
            raise TypeError("Artifact key requires a JSON object artifact.")
        return value[self.key]


Artifacts: TypeAlias = dict[str, ArtifactConfig]


class ArtifactsConfig(Config):
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def validate_mapping_input(cls, value: object) -> object:
        if isinstance(value, ArtifactsConfig):
            return value
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("ArtifactsConfig must be a mapping.")
        for name, source in value.items():
            if not isinstance(name, str):
                raise TypeError("ArtifactsConfig keys must be strings.")
            validate_serializable_value(source, f"artifacts.{name}")
        return value

    @model_validator(mode="after")
    def validate_entries(self) -> Self:
        for name, source in self.raw_entries().items():
            if not isinstance(name, str):
                raise TypeError("ArtifactsConfig keys must be strings.")
            validate_serializable_value(source, f"artifacts.{name}")
            ArtifactConfig.model_validate(source)
        return self

    def raw_entries(self) -> dict[str, ConfigData | ArtifactConfig]:
        return cast(
            dict[str, ConfigData | ArtifactConfig], dict(self.model_extra or {})
        )

    def artifacts(self, field: str = "artifacts") -> Artifacts:
        artifacts: Artifacts = {}
        for name, source in self.raw_entries().items():
            validate_serializable_value(source, f"{field}.{name}")
            artifacts[name] = cast(
                ArtifactConfig, ArtifactConfig.model_validate(source)
            )
        return artifacts

    def data(self, field: str = "artifacts") -> ConfigData:
        return {
            name: artifact.data() for name, artifact in self.artifacts(field).items()
        }
