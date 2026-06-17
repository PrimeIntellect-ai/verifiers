from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class TextArenaMessage(BaseModel):
    content: str


class TextArenaAction(Action):
    message: str


class TextArenaObservation(Observation):
    prompt: str = ""
    messages: list[TextArenaMessage] = Field(default_factory=list)
