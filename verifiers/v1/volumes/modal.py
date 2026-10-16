"""Existing Modal Volumes attached natively to Modal sandboxes."""

from typing import Literal

from pydantic import Field

from verifiers.v1.volumes.base import BaseVolumeConfig, Volume


class ModalVolumeConfig(BaseVolumeConfig):
    """A named Volume in the active Modal environment, using host SDK credentials.

    All mounts require v2 Volumes. Writable mounts commit with `sync` before
    artifact handoff. Running readers must reload after another sandbox commits;
    new sandboxes see the committed state.
    Concurrent writes to the same file have last-writer-wins semantics.
    """

    type: Literal["modal"] = "modal"
    name: str = Field(min_length=1)


class ModalVolume(Volume[ModalVolumeConfig]):
    config: ModalVolumeConfig

    def __init__(self, config: ModalVolumeConfig) -> None:
        super().__init__(config)
        self.handle = None

    @property
    def mount(self) -> ModalVolumeConfig:
        if self.handle is None:
            raise RuntimeError("volume is not resolved; call start() before using it")
        return self.config.model_copy()

    async def start(self) -> None:
        import modal

        handle = modal.Volume.from_name(
            self.config.name,
            version=2,
        ).with_mount_options(read_only=self.config.read_only)
        await handle.hydrate.aio()
        self.handle = handle

    async def stop(self) -> None:
        # Named volumes outlive their sandboxes. Dropping this SDK handle neither
        # commits sandbox-local writes nor deletes the persistent volume.
        self.handle = None
