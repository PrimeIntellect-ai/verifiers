import os


class MissingKeyError(ValueError):
    """Required environment variable(s) not set."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        key_list = ", ".join(keys)
        msg = (
            f"Missing required environment variable(s): {key_list}\n\n"
            "To set these variables:\n"
            "  - Environments Hub CI: Add secrets on the environment's Settings page\n"
            '  - Hosted Training: Set env_file in your config (e.g. env_file = ["secrets.env"])\n'
            "  - Local: Export in your shell (e.g. export OPENAI_API_KEY=...)"
        )
        super().__init__(msg)


def ensure_keys(keys: list[str]) -> None:
    """Validate that required environment variables are set.

    Args:
        keys: List of environment variable names to check

    Raises:
        MissingEnvKeyError: If any keys are not set (lists all missing)
    """
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise MissingKeyError(missing)
