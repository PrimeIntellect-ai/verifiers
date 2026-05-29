from collections.abc import Mapping
from typing import cast

from verifiers.types import ClientConfig, SamplingArgs

from .config import Config
from .types import ConfigData, ConfigInputMap, ConfigMap
from .utils.config_utils import string_mapping


class ModelConfig(Config):
    name: str | None = None
    client: ClientConfig | str | None = None
    sampling_args: SamplingArgs = {}


def model_config_from_value(value: object = None) -> ModelConfig:
    if isinstance(value, ModelConfig):
        return value
    if isinstance(value, str):
        return ModelConfig(name=value)
    if isinstance(value, Mapping):
        return ModelConfig.model_validate(string_mapping(cast(ConfigInputMap, value)))
    if value is None:
        return ModelConfig()
    raise TypeError("model must be a string or mapping.")


def model_config_data(value: object = None) -> ConfigData:
    return cast(
        ConfigData, model_config_from_value(value).model_dump(exclude_none=True)
    )


def model_config_from_task(task: ConfigMap) -> ModelConfig:
    value = task.get("model")
    return model_config_from_value(value)
