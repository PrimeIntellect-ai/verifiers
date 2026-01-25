"""
Actor: A trainable entity in multi-agent environments.

Actors are registered to a Protocol and define the system prompt
used when making model calls.
"""
from dataclasses import dataclass


@dataclass
class Actor:
    """
    A trainable actor. Registered to Protocol.

    The system_prompt is used when this actor makes model calls.
    """

    id: str
    system_prompt: str = ""
