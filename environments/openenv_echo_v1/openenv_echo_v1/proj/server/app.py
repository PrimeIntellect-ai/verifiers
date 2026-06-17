from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from server.environment import EchoEnvironment

app = create_app(
    EchoEnvironment, CallToolAction, CallToolObservation, env_name="echo_env"
)
