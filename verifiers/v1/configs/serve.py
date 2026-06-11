"""The `ServeConfig`: the config the env-server CLI parses.

Inherits `EnvServerConfig` (taskset + harness + timeouts + turn/token limits + the worker
`pool`), so the swappable harness/runtime knobs are the same flags as the eval CLI
(`--taskset.id`, `--harness.id`, `--harness.runtime.type`, `--taskset.*`, `--pool.*`), and
adds only the CLI-specific serving knobs (bind address, verbose, dry-run).
"""

from pydantic import AliasChoices, Field

from verifiers.v1.env import EnvServerConfig


class ServeConfig(EnvServerConfig):
    address: str = Field(
        "tcp://127.0.0.1:5000", validation_alias=AliasChoices("address", "a")
    )
    """ZMQ address the ROUTER binds (and clients connect to)."""
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    """Log at debug level instead of info."""
    dry_run: bool = False
    """Resolve + validate the config and dump it, then exit."""
