"""Serve one resource-server class from the published NeMo Gym package."""

import os
import socket
from importlib import import_module
from pathlib import Path

import uvicorn
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from omegaconf import OmegaConf

# Managed Gym servers share the evaluator host and never need a routable bind.
HOST = "127.0.0.1"
PORT_FILE = Path("nemo_gym.port")


def main() -> None:
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, 0))
    port = sock.getsockname()[1]

    module_name, class_name = os.environ["NEMO_GYM_RESOURCE_SERVER"].split(":", 1)
    server_class = getattr(import_module(module_name), class_name)
    config_class = server_class.model_fields["config"].annotation
    name = module_name.split(".")[-2]
    server = server_class(
        config=config_class(name=name, host=HOST, port=port, entrypoint="app.py"),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host=HOST, port=11000),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    app = server.setup_webserver()
    server.setup_liveness(app)
    server.setup_exception_middleware(app)
    PORT_FILE.write_text(str(port), encoding="ascii")
    uvicorn.Server(uvicorn.Config(app, host=HOST, port=port)).run(sockets=[sock])


if __name__ == "__main__":
    main()
