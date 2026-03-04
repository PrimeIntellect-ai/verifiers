# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application entrypoint for the TextArena environment."""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

try:
    # When running as installed package
    from textarena_env.models import TextArenaAction, TextArenaObservation
    from textarena_env.server.environment import TextArenaEnvironment
except ImportError:
    # When running uvicorn directly from textarena_env/
    from models import TextArenaAction, TextArenaObservation
    from .environment import TextArenaEnvironment


def _parse_env_kwargs(prefix: str = "TEXTARENA_KW_") -> dict[str, str]:
    """Collect arbitrary environment kwargs from the process environment."""

    env_kwargs: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            env_key = key[len(prefix) :].lower()
            env_kwargs[env_key] = value
    return env_kwargs


env_id = os.getenv("TEXTARENA_ENV_ID", "Wordle-v0")
num_players = int(os.getenv("TEXTARENA_NUM_PLAYERS", "1"))
max_turns_env = os.getenv("TEXTARENA_MAX_TURNS")
max_turns = int(max_turns_env) if max_turns_env is not None else None
download_nltk = os.getenv("TEXTARENA_DOWNLOAD_NLTK", "1") in {"1", "true", "True"}

extra_kwargs = _parse_env_kwargs()


# Factory function to create TextArenaEnvironment instances
def create_textarena_environment():
    """Factory function that creates TextArenaEnvironment with config."""
    return TextArenaEnvironment(
        env_id=env_id,
        num_players=num_players,
        max_turns=max_turns,
        download_nltk=download_nltk,
        env_kwargs=extra_kwargs,
    )


# Create the FastAPI app
# Pass the factory function instead of an instance for WebSocket session support
app = create_app(
    create_textarena_environment,
    TextArenaAction,
    TextArenaObservation,
    env_name="textarena_env",
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m textarena_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn textarena_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
