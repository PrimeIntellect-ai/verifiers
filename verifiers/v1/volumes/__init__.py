"""Existing bucket storage with a lifetime independent of Docker runtimes.

Provision the volume on the Linux container engine's host, then share its
serializable ``BindMount`` with the agent and grader::

    from verifiers.v1 import DockerConfig, S3VolumeConfig, provision_volume

    async with provision_volume(S3VolumeConfig(bucket="my-data")) as volume:
        config = DockerConfig(mounts={"/data": volume.mount})
        # Start and stop every consuming runtime inside this context.

S3 needs ``mount-s3``; Hugging Face Storage Buckets need ``hf-mount-fuse``.
Both require FUSE and ``user_allow_other`` in /etc/fuse.conf. Credentials belong
to the host client and are never copied into the runtime config. Bucket traffic
is host traffic, independent of the agent's network policy. The host filesystem
permissions must permit the container's execution user to access the mount.

Docker Desktop and remote daemons cannot attach an evaluator-local FUSE mount.
Volumes default to read-only. Writable buckets retain their client's filesystem
semantics: close/fsync and check errors before grading, share one attachment,
and use separate bucket prefixes when rollout outputs must be isolated.
Stopping a volume releases its attachment without deleting bucket contents.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from pydantic import Field

from verifiers.v1.volumes.base import BaseVolumeConfig, Volume
from verifiers.v1.volumes.huggingface import HuggingFaceVolume, HuggingFaceVolumeConfig
from verifiers.v1.volumes.s3 import S3Volume, S3VolumeConfig

VolumeConfig = Annotated[
    S3VolumeConfig | HuggingFaceVolumeConfig,
    Field(discriminator="type"),
]


def make_volume(config: VolumeConfig) -> Volume:
    return {"s3": S3Volume, "huggingface": HuggingFaceVolume}[config.type](config)


@asynccontextmanager
async def provision_volume(config: VolumeConfig) -> AsyncIterator[Volume]:
    """Attach existing storage, then detach after every consuming runtime exits."""
    volume = make_volume(config)
    try:
        await volume.start()
        yield volume
    finally:
        await volume.stop()


__all__ = [
    "BaseVolumeConfig",
    "HuggingFaceVolume",
    "HuggingFaceVolumeConfig",
    "S3Volume",
    "S3VolumeConfig",
    "Volume",
    "VolumeConfig",
    "make_volume",
    "provision_volume",
]
