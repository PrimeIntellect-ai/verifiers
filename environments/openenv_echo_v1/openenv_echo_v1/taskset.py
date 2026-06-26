"""OpenEnv's official Echo image as a zero-config example taskset."""

import verifiers.v1 as vf
from verifiers.v1.tasksets.openenv import OpenEnvConfig, OpenEnvTaskset

ECHO_IMAGE = (
    "ghcr.io/meta-pytorch/openenv-echo-env@"
    "sha256:56c55669c00b23a6af6adbcd8dd1fb5da3a276aec186b5c46cb4abeb708afa9c"
)


class OpenEnvEchoConfig(OpenEnvConfig):
    image: str = ECHO_IMAGE
    prompt: str = (
        'Call the echo_message tool with the message "Hello, World!", then return '
        "the echoed text."
    )
    resources: vf.TaskResources = vf.TaskResources(cpu=2, memory=4, disk=10)


class OpenEnvEchoTaskset(OpenEnvTaskset, vf.Taskset[vf.Task, OpenEnvEchoConfig]):
    pass
