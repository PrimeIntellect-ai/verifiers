"""Serve one resource-server class from the published NeMo Gym package."""

import os
import secrets
import socket
from importlib import import_module
from pathlib import Path

import uvicorn
from fastapi.responses import JSONResponse
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from omegaconf import OmegaConf


def main() -> None:
    # Bind only the sandbox's routed interface. UDP connect selects a local
    # address without sending traffic to this documentation-only destination.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
        route.connect(("192.0.2.1", 9))
        host = route.getsockname()[0]
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, int(os.environ.get("NEMO_GYM_PORT", "0"))))
    port = sock.getsockname()[1]

    module_name, class_name = os.environ["NEMO_GYM_RESOURCE_SERVER"].split(":", 1)
    server_class = getattr(import_module(module_name), class_name)
    config_class = server_class.model_fields["config"].annotation
    name = module_name.rsplit(".", 2)[-2] if "." in module_name else module_name
    server = server_class(
        config=config_class(name=name, host=host, port=port, entrypoint="app.py"),
        server_client=ServerClient(
            head_server_config=BaseServerConfig(host="127.0.0.1", port=11000),
            global_config_dict=OmegaConf.create({}),
        ),
    )
    app = server.setup_webserver()
    server.setup_liveness(app)
    server.setup_exception_middleware(app)
    token = os.environ["NEMO_GYM_TOKEN"]

    @app.middleware("http")
    async def authenticate(request, call_next):
        supplied = request.headers.get("X-Verifiers-Nemo-Token", "")
        if not secrets.compare_digest(supplied, token):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

    Path("/tmp/nemo_gym.port").write_text(str(port), encoding="ascii")
    uvicorn.Server(uvicorn.Config(app, host=host, port=port)).run(sockets=[sock])


if __name__ == "__main__":
    main()
