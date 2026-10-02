"""Host-mounted storage, owned independently of the runtimes using it."""

import asyncio
import atexit
import re
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Generic, TypeVar

from pydantic_config import BaseConfig

from verifiers.v1.configs.runtime import BindMount
from verifiers.v1.utils.aio import run_shielded


class BaseVolumeConfig(BaseConfig):
    read_only: bool = True
    """Whether the storage client permits writes. Bucket permissions still apply."""


MountConfig = TypeVar("MountConfig", bound=BaseConfig)


class Volume(ABC, Generic[MountConfig]):
    """Storage with a lifetime independent of the runtimes attaching to it."""

    def __init__(self, config: BaseVolumeConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def mount(self) -> MountConfig:
        """Serializable attachment configuration for a compatible runtime."""

    @abstractmethod
    async def start(self) -> None:
        """Resolve the storage and acquire any resources needed to attach it."""

    @abstractmethod
    async def stop(self) -> None:
        """Release attachment resources without deleting persistent storage."""


class HostVolume(Volume[BindMount]):
    """One host attachment to an existing bucket, shared by any number of runtimes.

    Provision on the Linux container engine's host and pass the same `mount` to the agent
    and grader. Stop every consumer before stopping the volume. A remote daemon or
    Docker Desktop VM cannot use an evaluator-local mount through this interface.
    Credentials are read by the host storage client, never added to runtime config.
    Bucket I/O is host I/O and is independent of the agent's network policy.
    """

    def __init__(self, config: BaseVolumeConfig) -> None:
        super().__init__(config)
        self._path: Path | None = None
        self._process: subprocess.Popen | None = None
        self._log: BinaryIO | None = None
        self._ready = False

    @property
    def mount(self) -> BindMount:
        """Bind this live attachment into a Docker, Podman, or Apptainer runtime."""
        if (
            self._path is None
            or self._process is None
            or self._process.poll() is not None
            or not self._is_mounted()
        ):
            raise RuntimeError("volume is not mounted; call start() before using it")
        return BindMount(source=str(self._path), read_only=self.config.read_only)

    def _is_mounted(self) -> bool:
        # stat() can fail on a disconnected FUSE mount after a client crash. The
        # kernel mount table still identifies it so cleanup can unmount it.
        target = re.sub(r"[ \t\n\\]", lambda m: f"\\{ord(m[0]):03o}", str(self._path))
        return any(
            line.split()[4] == target
            for line in Path("/proc/self/mountinfo").read_text().splitlines()
        )

    async def start(self) -> None:
        """Mount once, retaining ownership even if the caller is cancelled."""
        await run_shielded(asyncio.to_thread(self._start))

    def _start(self) -> None:
        if self._path is not None:
            raise RuntimeError("volume already started; stop it before restarting")
        if sys.platform != "linux":
            raise ValueError("bucket volumes require FUSE on the Linux container host")
        self._path = Path(tempfile.mkdtemp(prefix="vf-volume-")).resolve()
        atexit.register(self._cleanup_at_exit)
        command = self._mount_command(self._path)
        if not shutil.which(command[0]):
            raise FileNotFoundError(f"install {command[0]} on the container host")
        self._log = tempfile.TemporaryFile()  # noqa: SIM115 - owned until cleanup()
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 60
        while self._process.poll() is None:
            if self._is_mounted():
                self._ready = True
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{command[0]} did not mount {self._path} within 60s"
                )
            time.sleep(0.05)
        self._log.seek(0)
        raise RuntimeError(
            f"{command[0]} failed to mount: {self._log.read().decode(errors='replace').strip()}"
        )

    async def stop(self) -> None:
        """Unmount after consumers exit; uploaded bucket data outlives this handle."""
        await run_shielded(asyncio.to_thread(self.cleanup))

    def cleanup(self) -> None:
        """Detach without deleting bucket contents; a failed detach can be retried."""
        if self._path is None:
            return
        mounted = self._is_mounted()
        if mounted:
            # A busy mount is an ownership error: leave it and the client running so
            # the owner can stop the remaining consumers and retry, without data loss.
            result = subprocess.run(
                [shutil.which("fusermount3") or "fusermount", "-u", str(self._path)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"could not unmount {self._path}: {result.stderr.strip()}"
                )
        error = ""
        if self._process is not None:
            # Foreground clients normally exit when unmounted. A failed startup
            # still needs terminating even if it never reached mount readiness.
            if not mounted and self._process.poll() is None:
                self._process.terminate()
            code = self._process.wait(timeout=60)
            if self._ready and code != 0:
                assert self._log is not None
                self._log.seek(0)
                error = f"storage client exited with code {code}: {self._log.read().decode(errors='replace').strip()}"
            self._process = None
        if self._log is not None:
            self._log.close()
            self._log = None
        # Never recursively remove a mount: rmdir can only remove our empty directory.
        self._path.rmdir()
        self._path = None
        self._ready = False
        atexit.unregister(self._cleanup_at_exit)
        if error:
            raise RuntimeError(error)

    def _cleanup_at_exit(self) -> None:
        # Runtime cleanup may have been registered before this volume. Explicitly
        # stop consumers first instead of depending on atexit's registration order.
        from verifiers.v1.runtimes.base import cleanup_at_exit

        cleanup_at_exit()
        self.cleanup()

    @abstractmethod
    def _mount_command(self, path: Path) -> list[str]:
        pass
