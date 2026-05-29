from typing import TYPE_CHECKING, cast

from verifiers.types import ClientConfig, SamplingArgs

from .config import Config
from .types import ConfigData, ModelClient
from .utils.config_utils import resolve_config_object, string_mapping

if TYPE_CHECKING:
    from .task import Task


class ModelConfig(Config):
    name: str | None = None
    client: ClientConfig | str | None = None
    sampling_args: SamplingArgs = {}

    def client_object(self) -> ModelClient | None:
        if self.client is None:
            return None
        client = resolve_config_object(self.client)
        if isinstance(client, ClientConfig):
            return client
        from verifiers.clients import Client

        if isinstance(client, Client):
            return client
        raise TypeError("model.client must resolve to a Client or ClientConfig.")


def model_config_from_value(value: object = None) -> ModelConfig:
    if isinstance(value, ModelConfig):
        return value
    if isinstance(value, str):
        return ModelConfig(name=value)
    if isinstance(value, dict):
        return ModelConfig.model_validate(string_mapping(value))
    if value is None:
        return ModelConfig()
    raise TypeError("model must be a string or mapping.")


def model_config_data(value: object = None) -> ConfigData:
    return cast(
        ConfigData, model_config_from_value(value).model_dump(exclude_none=True)
    )


def model_config_from_task(task: "Task") -> ModelConfig:
    value = task.get("model")
    return model_config_from_value(value)
