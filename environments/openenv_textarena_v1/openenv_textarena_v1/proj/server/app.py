from openenv.core.env_server.http_server import create_app

from server.environment import TextArenaEnvironment
from server.models import TextArenaAction, TextArenaObservation

app = create_app(
    TextArenaEnvironment,
    TextArenaAction,
    TextArenaObservation,
    env_name="textarena_env",
)
