"""Environment-server CLI configuration."""

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator

from verifiers.v1.env import EnvServerConfig
from verifiers.v1.topology import TopologyConfig


class ServeConfig(EnvServerConfig):
    topology: SerializeAsAny[TopologyConfig] | None = None
    """An explicit topology; otherwise taskset + harness lowers to single-agent."""
    address: str = Field("tcp://127.0.0.1:5000", validation_alias=AliasChoices("address", "a"))
    """ZMQ address the ROUTER binds (and clients connect to)."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of info."""
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_topology(cls, data):
        if isinstance(data, dict) and data.get("topology"):
            if data.get("harness"):
                raise ValueError("`--harness.*` is ignored under a topology; configure each agent's harness")
            from verifiers.v1.loaders import narrow_plugin_field, topology_config_type

            narrow_plugin_field(data, "topology", topology_config_type)
        return data

    @model_validator(mode="after")
    def check_topology(self):
        if self.topology is not None and (self.taskset.id or self.id):
            raise ValueError("`--topology.id` replaces `--taskset.id` / `--id`; choose seeds under the topology")
        return self
