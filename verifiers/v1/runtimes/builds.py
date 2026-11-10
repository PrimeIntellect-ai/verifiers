"""Prepare a Dockerfile image before provisioning an ordinary isolated verifier."""

import json
import os
import tempfile
from pathlib import Path

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, RuntimeConfig
from verifiers.v1.runtimes.backends.docker import cli
from verifiers.v1.runtimes.compose import DOCKER_ENV, ComposeSpec


async def prepare_verifier_image(
    config: RuntimeConfig, spec: ComposeSpec, env: dict[str, str]
) -> tuple[RuntimeConfig, dict[str, str], str | None]:
    """Build current inputs, preserving the image's workdir, environment and user."""
    if spec.file is not None:
        raise SandboxError("Verifier builds require a Dockerfile, not live services")
    if spec.image_override:
        return config, env, None
    if isinstance(config, PrimeConfig):
        from verifiers.v1.runtimes.native_compose import prepare_native

        service = (await prepare_native(config, spec, env))[spec.workspace]
        return service.config, {**service.image_env, **env}, service.user
    if not isinstance(config, DockerConfig):
        raise SandboxError(f"Verifier builds are not supported by {config.type}")
    docker_env = {key: os.environ[key] for key in DOCKER_ENV if key in os.environ}
    with tempfile.TemporaryDirectory(prefix="vf-verifier-build-") as directory:
        image_file = Path(directory) / "image"
        result = await cli(
            "docker",
            "build",
            "--iidfile",
            str(image_file),
            spec.directory,
            env=docker_env,
        )
        if result.exit_code:
            raise SandboxError(f"Verifier image build failed: {result.stderr}")
        image = image_file.read_text().strip()
    result = await cli("docker", "image", "inspect", image, env=docker_env)
    if result.exit_code:
        raise SandboxError(f"Verifier image inspection failed: {result.stderr}")
    metadata = json.loads(result.stdout)[0]["Config"]
    return (
        config.model_copy(
            update={
                "image": image,
                "workdir": config.workdir or metadata.get("WorkingDir") or "/",
            }
        ),
        env,
        metadata.get("User") or None,
    )
