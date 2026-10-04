"""Storage volumes with lifetimes independent of agent and grader runtimes.

On a Linux container host with FUSE and the provider's mount client installed
(`mount-s3` or `hf-mount-fuse`), allow other users through `user_allow_other` in
`/etc/fuse.conf` so container users can access the mount. Unix permissions still
apply. Mounts share the client's host credentials across their consumers::

    from verifiers.v1 import DockerConfig, S3VolumeConfig, provision_volume
    from verifiers.v1.runtimes import provision_runtime

    async with provision_volume(S3VolumeConfig(bucket="my-data")) as volume:
        config = DockerConfig(mounts={"/data": volume.mount})
        async with provision_runtime(config) as agent:
            ...
        async with provision_runtime(config) as grader:
            ...

Both runtimes use the same attachment. The bucket must already exist; Verifiers
never creates or deletes it. Mounts are lazy remote filesystems, not downloaded
snapshots. Use separate bucket prefixes when rollouts must not share their writes.
Prime and Modal do not accept these host bind mounts. Docker Desktop and remote
container engines require provisioning on their Linux host, not on the evaluator.

Modal uses native attachments, without host FUSE clients::

    from verifiers.v1 import ModalConfig, ModalVolumeConfig

    config = ModalConfig(mounts={"/data": ModalVolumeConfig(name="my-data")})

`provision_volume(ModalVolumeConfig(...))` also provides a reusable `volume.mount`.
The named volume must already exist and use v2. Writable mounts
commit before artifact collection and sandbox teardown. Use
`await writer.commit_volumes()` before starting another reader while the writer
is still running; an existing reader needs `await reader.reload_volumes()` with
all its volume files closed. Stopping a volume handle does not delete its data.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from pydantic import Field

from verifiers.v1.volumes.base import BaseVolumeConfig, Volume
from verifiers.v1.volumes.huggingface import HuggingFaceVolume, HuggingFaceVolumeConfig
from verifiers.v1.volumes.modal import ModalVolume, ModalVolumeConfig
from verifiers.v1.volumes.s3 import S3Volume, S3VolumeConfig

VolumeConfig = Annotated[
    S3VolumeConfig | HuggingFaceVolumeConfig | ModalVolumeConfig,
    Field(discriminator="type"),
]


def make_volume(config: VolumeConfig) -> Volume:
    return {
        "s3": S3Volume,
        "huggingface": HuggingFaceVolume,
        "modal": ModalVolume,
    }[config.type](config)


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
    "ModalVolume",
    "ModalVolumeConfig",
    "S3Volume",
    "S3VolumeConfig",
    "Volume",
    "VolumeConfig",
    "make_volume",
    "provision_volume",
]
