"""Local Apptainer runtime: an unprivileged instance sharing the host network."""

import asyncio
import contextlib
import hashlib
import logging
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import ClassVar, Literal

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import BaseRuntimeInfo, parse_gpu
from verifiers.v1.runtimes.container import ContainerConfig, ContainerRuntime, cli
from verifiers.v1.utils.paths import CACHE_DIR

logger = logging.getLogger(__name__)

_ROOT = CACHE_DIR / "runtimes" / "apptainer"


class ApptainerConfig(ContainerConfig):
    type: Literal["apptainer"] = "apptainer"
    """Apptainer has no egress policy: instances run unprivileged on the host network,
    as on HPC clusters without Docker. `image` is a Docker reference (pulled to a SIF
    once per reference), any `scheme://` URI Apptainer can pull, or a local SIF path."""


class ApptainerRuntimeInfo(ApptainerConfig, BaseRuntimeInfo):
    pass


class ApptainerRuntime(ContainerRuntime):
    _pulls: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(self, config: ApptainerConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = ApptainerRuntimeInfo(**config.model_dump())
        self._dir: Path | None = None  # host backing for the workspace, /tmp and $HOME
        self._stopped = False

    async def _exec(self, env: dict[str, str], *, stdin: bool = False) -> list[str]:
        # `--env` is a comma-separated map flag, so values pass through `env` inside.
        return [
            "apptainer",
            "exec",
            "--cleanenv",
            "--no-eval",
            "--cwd",
            self.config.workdir,
            f"instance://{self.name}",
            "env",
            *(f"{key}={value}" for key, value in env.items()),
        ]

    async def start(self) -> None:
        try:
            version = await cli("apptainer", "version")
        except FileNotFoundError as e:
            raise RuntimeError(
                "apptainer runtime selected but the `apptainer` CLI is not installed"
            ) from e
        if version.exit_code != 0:
            detail = (version.stderr or version.stdout).strip()
            raise RuntimeError(
                f"apptainer runtime selected but Apptainer is not usable: {detail}"
            )
        self._dir = _ROOT / "instances" / self.name
        (self._dir / "workspace").mkdir(parents=True)
        (self._dir / "session").mkdir()
        limits: list[str] = []
        if self.config.cpu is not None:
            limits += ["--cpus", str(self.config.cpu)]
        if self.config.memory is not None:
            limits += ["--memory", f"{self.config.memory}g"]
        if parse_gpu(self.config.gpu)[1]:
            limits += ["--nv"]
        started = await cli(
            "apptainer",
            "instance",
            "start",
            "--cleanenv",
            "--containall",
            "--no-eval",
            # A contained /tmp and $HOME live in tmpfs unless a workdir backs them.
            "--workdir",
            str(self._dir / "session"),
            "--writable-tmpfs",
            "--bind",
            f"{self._dir / 'workspace'}:{self.config.workdir}",
            *limits,
            await self._image(),
            self.name,
        )
        if started.exit_code != 0:
            raise SandboxError(
                f"apptainer instance start failed: {started.stderr.strip()}"
            )
        self.info.id = self.name
        logger.info(
            "apptainer: started instance %s (image=%s)", self.name, self.config.image
        )

    async def _image(self) -> str:
        """The SIF to run: a local file as is, else the reference pulled once into the
        cache, like a container engine keeping a pulled image."""
        local = Path(self.config.image).expanduser()
        if local.is_file():
            return str(local)
        image = self.config.image
        ref = image if "://" in image else f"docker://{image}"
        sif = _ROOT / "images" / f"{hashlib.sha256(ref.encode()).hexdigest()}.sif"
        async with self._pulls.setdefault(ref, asyncio.Lock()):
            if not sif.exists():
                sif.parent.mkdir(parents=True, exist_ok=True)
                # Pull to a unique path and publish it atomically, so concurrent
                # workers never read a half-written SIF.
                tmp = sif.with_name(f"{uuid.uuid4().hex}.sif")
                pulled = await cli("apptainer", "pull", str(tmp), ref)
                if pulled.exit_code != 0:
                    raise SandboxError(
                        f"apptainer pull {ref} failed: {pulled.stderr.strip()}"
                    )
                tmp.replace(sif)
        return str(sif)

    def cleanup(self) -> None:
        if self._dir is None or self._stopped:
            return
        self._stopped = True
        logger.debug("apptainer: stopping instance %s", self.name)
        with contextlib.suppress(Exception):
            subprocess.run(
                ["apptainer", "instance", "stop", "--force", self.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
        shutil.rmtree(self._dir, ignore_errors=True)
