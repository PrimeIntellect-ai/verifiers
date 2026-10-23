"""Model pools: several deployments an agent may use, each behind its own request gate.

A session picks a deployment at its first model request and keeps it (see
`model_routing`): deployments are tried in random order and the first whose gate has a
free slot wins, so sessions fill whatever has room, under every deployment's cap.
"""

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.configs.client import ClientConfig, TrainClientConfig


class ModelDeployment(BaseConfig):
    """One place a model is served: its endpoint (`client`, which must set a
    `request_gate`, its capacity) and the model id it is asked for there."""

    id: str = Field(min_length=1)
    """Names the deployment in the trace (`ModelCall.deployment`) and scopes its gate."""
    model: str = Field(min_length=1)
    client: ClientConfig

    @model_validator(mode="after")
    def _gated(self) -> "ModelDeployment":
        gate = self.client.request_gate
        if gate is None:
            raise ValueError(
                f"deployment {self.id!r} needs a client.request_gate: its capacity"
            )
        if gate.scope is None:  # deployments behind one URL and key keep their own caps
            self.client = self.client.model_copy(
                update={"request_gate": gate.model_copy(update={"scope": self.id})}
            )
        return self


class ModelPoolConfig(BaseConfig):
    deployments: list[ModelDeployment] = Field(min_length=1)
    same_model: bool = False
    """Require every deployment to serve the same model id: replicas of one model and
    revision, so results from any of them are interchangeable (e.g. calibration labels,
    which must not mix models)."""

    @model_validator(mode="after")
    def _consistent(self) -> "ModelPoolConfig":
        if len({item.id for item in self.deployments}) != len(self.deployments):
            raise ValueError("deployment ids must be unique within a model pool")
        models = {item.model for item in self.deployments}
        if self.same_model and len(models) > 1:
            raise ValueError(f"same_model pool serves several models: {sorted(models)}")
        training = [
            item
            for item in self.deployments
            if isinstance(item.client, TrainClientConfig)
        ]
        if training and (
            len(training) != len(self.deployments)
            or len(models) != 1
            or any(
                (item.client.renderer, item.client.renderer_model_name)
                != (training[0].client.renderer, training[0].client.renderer_model_name)
                for item in training
            )
        ):
            raise ValueError(
                "a training pool needs the same model and renderer on every deployment"
            )
        return self
