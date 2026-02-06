# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the TextArena Environment.

This module defines the action, observation, and state models for interacting
with TextArena game environments (e.g., Wordle-v0).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


from openenv.core.env_server.types import Action, Observation, State


class TextArenaMessage(BaseModel):
    """Single message observed by a player."""

    sender_id: int
    content: str
    category: str


class TextArenaAction(Action):
    """Action issued by the agent for TextArena games."""

    message: str


class TextArenaObservation(Observation):
    """Observation returned from any TextArena game."""

    prompt: str
    messages: List[TextArenaMessage] = Field(default_factory=list)
    current_player_id: int = 0
    legal_players: List[int] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)


class TextArenaState(State):
    """Structured state snapshot for the server."""

    episode_id: Optional[str] = None
    step_count: int = 0
    env_id: str
    num_players: int
    max_turns: Optional[int] = None
    turn: int = 0
    last_reward: float = 0.0
    last_info: Dict[str, Any] = Field(default_factory=dict)
    raw_state: Dict[str, Any] = Field(default_factory=dict)
