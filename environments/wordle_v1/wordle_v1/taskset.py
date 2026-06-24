"""wordle-v1 — the textarena taskset pinned to Wordle (example env).

A thin wrapper over `textarena`: pins `game` to "Wordle-v0", so `wordle-v1` is a
zero-config Wordle env. Everything else — the user simulator, seed-based task generation,
and game-authoritative scoring — is inherited unchanged.
"""

from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.tasksets.textarena import (
    TextArenaConfig,
    TextArenaState,
    TextArenaTask,
    TextArenaTaskset,
)


class WordleConfig(TextArenaConfig):
    game: Literal["Wordle-v0"] = "Wordle-v0"


class WordleTaskset(
    TextArenaTaskset, vf.Taskset[TextArenaTask, WordleConfig, TextArenaState]
):
    pass
