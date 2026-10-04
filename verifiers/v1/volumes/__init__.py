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

    from verifiers.v1 import (
        HuggingFaceVolumeConfig, ModalConfig, ModalVolumeConfig, S3VolumeConfig,
    )

    config = ModalConfig(mounts={
        "/data": ModalVolumeConfig(name="my-data"),
        "/outputs": S3VolumeConfig(
            bucket="my-outputs", prefix="run-123/", read_only=False,
            modal_secret="aws-storage",
        ),
        "/hf": HuggingFaceVolumeConfig(
            bucket="my-org/my-data", modal_secret="hf-storage",
        ),
    })

Bucket mounts resolve on Modal's host through CloudBucketMount. Named Secrets
are used only by the storage mount, not injected into sandbox process environments;
evaluator credentials are not forwarded. Private S3 mounts require a Secret with
AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, plus AWS_SESSION_TOKEN when needed.
Set AWS_REGION in that Secret when region detection is unsuitable. The S3 config's
host region/profile options are rejected on Modal; public buckets use anonymous=True.

Hugging Face mounts use https://s3.hf.co/<namespace> with path-style addressing.
Their Secret requires Hugging Face-generated S3 access keys under the same AWS
key names and AWS_REGION=us-east-1; HF_TOKEN cannot authenticate to this gateway.
Hugging Face mounts are read-only because Modal exposes no control for the
gateway's upload checksum requirements. See https://huggingface.co/docs/hub/storage-buckets-s3.

Close/fsync S3 files before handing them to a grader; bucket mounts do not use
Modal's snapshot commit/reload operations. Freshness follows the bucket client's
cache, so use a fresh grader sandbox after the writer closes its files. Mount I/O
runs outside sandbox execution networking and grants access under the configured
storage credentials. Use bucket permissions to constrain that access.

`provision_volume(ModalVolumeConfig(...))` also provides a reusable `volume.mount`.
Named Modal volumes must already exist and use v2. Writable Modal volumes
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
