# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "nemo-gym==0.4.0",
# ]
# ///
"""Launch the upstream weather resources server without NeMo's agent stack."""

import os

import uvicorn
from omegaconf import OmegaConf

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from resources_servers.example_mcp_weather.app import (
    ExampleMCPWeatherResourcesServer,
    ExampleMCPWeatherResourcesServerConfig,
)

HOST = os.environ.get("NEMO_GYM_HOST", "127.0.0.1")
PORT = int(os.environ.get("NEMO_GYM_PORT", "8000"))


def main() -> None:
    server = ExampleMCPWeatherResourcesServer(
        config=ExampleMCPWeatherResourcesServerConfig(
            name="example_mcp_weather",
            host=HOST,
            port=PORT,
            entrypoint="app.py",
        ),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host=HOST, port=11000),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    app = server.setup_webserver()
    server.setup_liveness(app)
    server.setup_exception_middleware(app)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
