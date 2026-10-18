"""Eligible model deployments for one native agent session."""

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.client import ClientConfig, TrainClientConfig


class ModelDeployment(BaseConfig):
    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    client: ClientConfig

    @model_validator(mode="after")
    def checked(self):
        gate = self.client.request_gate
        if gate is None:
            raise ValueError("each pooled deployment needs a request_gate capacity")
        if gate.scope is None:
            self.client = self.client.model_copy(
                update={"request_gate": gate.model_copy(update={"scope": self.id})}
            )
        return self


class ModelPoolConfig(BaseConfig):
    deployments: list[ModelDeployment] = Field(min_length=1)

    @model_validator(mode="after")
    def checked(self):
        if len({item.id for item in self.deployments}) != len(self.deployments):
            raise ValueError("deployment ids must be unique within a model pool")
        training = [
            item
            for item in self.deployments
            if isinstance(item.client, TrainClientConfig)
        ]
        if training and (
            len(training) != len(self.deployments)
            or len({item.model for item in training}) != 1
            or any(
                item.client.renderer != training[0].client.renderer for item in training
            )
            or any(
                item.client.renderer_model_name
                != training[0].client.renderer_model_name
                for item in training
            )
        ):
            raise ValueError(
                "training pools require the same model and renderer on every deployment"
            )
        return self
