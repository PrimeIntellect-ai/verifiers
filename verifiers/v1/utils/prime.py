from prime_sandboxes import Config


def prime_config() -> Config:
    """The active Prime CLI config, resolved by the prime SDK exactly as `prime` does.

    `PRIME_*` env vars, then `--context`/`$PRIME_CONTEXT`, then the directory's
    `.prime/context.json` (e.g. `prime switch --local`), then `~/.prime/config.json`.
    Raises `ValueError` when a selected context is missing or invalid: falling back
    to the global config would bill the wrong account.
    """
    return Config()


def load_prime_config() -> dict:
    """The active Prime CLI config values without `PRIME_*` env overrides."""
    return prime_config().config


def ensure_prime_auth() -> None:
    """Exit when no Prime API key is configured (`prime login` or `$PRIME_API_KEY`)."""
    try:
        api_key = prime_config().api_key
    except ValueError as e:
        raise SystemExit(f"invalid prime config: {e}") from e
    if api_key:
        return
    raise SystemExit(
        "not authenticated with prime - run `prime login` or set $PRIME_API_KEY"
    )
