"""How an env is served: the worker pool, where it binds, and each worker's bound.

Serving is its own axis. `[env]` says what runs; this says how it's hosted, so an
eval, a `serve` process, and a trainer configure it identically instead of each
flattening pool knobs onto its own config."""

from typing import Annotated, Literal

from pydantic import Field
from pydantic_config import BaseConfig


class StaticPoolConfig(BaseConfig):
    """Fixed env-server pool: pre-spawn `num_workers` workers up front."""

    type: Literal["static"] = "static"
    num_workers: int = Field(4, ge=1)
    """Worker processes to pre-spawn (1 = a single in-process server, no pool)."""


class ElasticPoolConfig(BaseConfig):
    """Elastic env-server pool: start at one worker and scale up on demand."""

    type: Literal["elastic"] = "elastic"
    max_workers: int | None = None
    """Upper bound on workers (None = unbounded)."""
    multiplex: int = Field(128, ge=1)
    """Episodes per worker for the scale-up trigger: add a worker once in-flight episodes
    reach 90% of `workers * multiplex`."""


# Discriminated on `type` so the CLI selects with `--serve.pool.type static|elastic`.
PoolConfig = Annotated[
    StaticPoolConfig | ElasticPoolConfig, Field(discriminator="type")
]


class ServingConfig(BaseConfig):
    """The `[serve]` block: the worker pool, the ZMQ address, and each worker's
    episode bound. Read by whoever hosts the env — the `serve` CLI, a server-backed
    eval, a trainer's orchestrator."""

    pool: PoolConfig = Field(default_factory=ElasticPoolConfig)
    """Worker-pool sizing. `elastic` (default) starts at one worker and scales up on
    demand; `static` pre-spawns a fixed `num_workers`."""
    address: str = "tcp://127.0.0.1:5000"
    """ZMQ address the ROUTER binds (and clients connect to)."""
    max_concurrent: int | None = Field(None, ge=1)
    """Episodes in flight per worker (None = take the run's own bound, e.g. an eval's
    `--max-concurrent`). Pin it to hold a worker below what the run asks for; how many
    agent runs one episode carries is the env's own
    (`--env.max-concurrent-agents`, one at a time by default)."""


def pool_serve_kwargs(pool: StaticPoolConfig | ElasticPoolConfig) -> dict:
    """Unpack a pool config into `serve_env` kwargs (`max_workers` / `multiplex` / `elastic`)."""
    if isinstance(pool, ElasticPoolConfig):
        return {
            "max_workers": pool.max_workers,
            "multiplex": pool.multiplex,
            "elastic": True,
        }
    return {"max_workers": pool.num_workers, "elastic": False}
