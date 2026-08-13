import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_prime_config() -> dict:
    """The user's `~/.prime/config.json` (`prime login`), `{}` when absent/invalid."""
    try:
        config_file = Path.home() / ".prime" / "config.json"
        if config_file.exists():
            data = json.loads(config_file.read_text())
            if isinstance(data, dict):
                return data
            logger.warning("Invalid prime config: expected dict")
    except (RuntimeError, json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load prime config: {e}")
    return {}


def ensure_prime_auth() -> None:
    """Exit when no Prime API key is configured (`$PRIME_API_KEY` or `prime login`)."""
    if os.getenv("PRIME_API_KEY") or load_prime_config().get("api_key"):
        return
    raise SystemExit(
        "not authenticated with Prime Intellect - set $PRIME_API_KEY or run `prime login`"
    )
