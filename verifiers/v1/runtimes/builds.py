"""Prepare a Dockerfile image before provisioning an ordinary isolated verifier."""

import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes import DockerConfig, PrimeConfig, Runtime, RuntimeConfig
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

        # Inspect the image without provisioning any of the grader's resources.
        service = (
            await prepare_native(
                config.model_copy(update={"gpu": None, "mounts": {}}), spec, env
            )
        )[spec.workspace]
        return (
            service.config.model_copy(
                update={"gpu": config.gpu, "mounts": config.mounts}
            ),
            {**service.image_env, **env},
            service.user,
        )
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
                "workdir": config.workdir or metadata.get("WorkingDir") or "/app",
            }
        ),
        env,
        metadata.get("User") or None,
    )


async def prime_verifier_user(runtime: Runtime, user: str | None) -> str | None:
    """Represent an OCI user as a guest username for Prime's command API."""
    if user is None or (":" not in user and not user.isdecimal()):
        return user

    uid_spec, _, gid_spec = user.partition(":")
    uid_spec = uid_spec or "0"
    passwd = (await runtime.read("/etc/passwd")).decode(errors="replace").splitlines()
    accounts = [row for line in passwd if len(row := line.split(":")) == 7]
    if uid_spec.isdecimal():
        uid = str(int(uid_spec))
        account = next((row for row in accounts if row[2] == uid), None)
    else:
        account = next((row for row in accounts if row[0] == uid_spec), None)
        if account is None:
            raise SandboxError(f"Unknown verifier image user {uid_spec!r}")
        uid = account[2]

    if account and not gid_spec:
        return account[0]
    if gid_spec.isdecimal():
        gid = str(int(gid_spec))
    elif gid_spec:
        group = (await runtime.read("/etc/group")).decode(errors="replace").splitlines()
        groups = [row for line in group if len(row := line.split(":")) == 4]
        match = next((row for row in groups if row[0] == gid_spec), None)
        if match is None:
            raise SandboxError(f"Unknown verifier image group {gid_spec!r}")
        gid = match[2]
    else:
        # Docker defaults an unmapped numeric UID to group 0.
        gid = "0"

    # A group override discards the original account's supplementary groups.
    # The fresh verifier's alias has no entries in /etc/group.
    alias = f"vf-grader-{uuid4().hex[:16]}"
    home, shell = account[5:7] if account else ("/", "/bin/sh")
    entry = f"{alias}:x:{uid}:{gid}::{home or '/'}:{shell or '/bin/sh'}"
    result = await runtime.run(
        ["sh", "-c", 'printf "\\n%s\\n" "$1" >> /etc/passwd', "sh", entry], {}
    )
    if result.exit_code:
        raise SandboxError(
            f"Could not set verifier image user {user!r}: {result.stderr}"
        )
    return alias
