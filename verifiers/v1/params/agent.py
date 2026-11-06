"""What an `Agent` is built from: its config with the run's bindings filled in."""

from __future__ import annotations

from pydantic import SerializeAsAny

from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.client import ClientConfig, EvalClientConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.utils.generic import merge_defaults


class AgentParams(AgentConfig):
    """A seat's `AgentConfig` with every run binding resolved: the agent this trace
    can be reproduced with (`Agent(trace.agent.config)`)."""

    harness: SerializeAsAny[HarnessConfig]
    model: str
    client: ClientConfig
    sampling: SamplingConfig

    @classmethod
    def resolve(
        cls,
        config: AgentConfig,
        *,
        model: str | None = None,
        client: ClientConfig | None = None,
        sampling: SamplingConfig | None = None,
        harness: HarnessConfig | None = None,
    ) -> AgentParams:
        """`config` with what it leaves unset filled from the run: its own fields
        win, its sampling values merge over the run's, and `max_turn_tokens` caps
        `sampling.max_tokens`. An unpinned harness falls back to `harness`, else
        `bash`.

        The run's sampling belongs to the run's client: it can carry endpoint-specific
        keys (e.g. vLLM's `top_k`) that another provider rejects. A seat that pins its
        own `client` therefore starts from empty sampling."""
        from verifiers.v1.utils.loaders import harness_config_type

        model = config.model if config.model is not None else model
        if model is None:
            raise ValueError(
                "AgentConfig.model is unset; an Agent needs a pinned model "
                "(inside an env the run's own model fills it in)"
            )
        defaults = (
            sampling
            if sampling is not None and config.client is None
            else SamplingConfig()
        )
        overrides = (config.sampling or SamplingConfig()).model_dump(exclude_unset=True)
        merged = {**defaults.model_dump(), **merge_defaults(defaults, overrides)}
        if config.max_turn_tokens is not None:
            max_tokens = merged.get("max_tokens")
            merged["max_tokens"] = min(
                max_tokens or config.max_turn_tokens, config.max_turn_tokens
            )
        bindings = {
            "harness": config.harness
            or harness
            or harness_config_type("bash")(id="bash"),
            "model": model,
            "client": config.client or client or EvalClientConfig(),
            "sampling": SamplingConfig.model_validate(merged),
        }
        # Already-validated values: constructing keeps plugin config subclasses as they are.
        return cls.model_construct(
            config.model_fields_set | bindings.keys(), **{**dict(config), **bindings}
        )
