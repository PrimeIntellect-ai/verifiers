"""Hugging Face Storage Buckets through the official hf-mount client."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from verifiers.v1.volumes.base import BaseVolumeConfig, HostVolume


class HuggingFaceVolumeConfig(BaseVolumeConfig):
    """An existing Storage Bucket, not a model or dataset repository.

    Host mounts require `hf-mount-fuse` and FUSE on Linux; authentication uses
    `HF_TOKEN` in the host environment. Host writes use the client's streaming mode:
    write sequentially and close/fsync before grading. Existing files can be
    replaced with O_TRUNC; random writes are unsupported. Separate mounts may see stale data and have no
    shared file locks; share one attachment with the agent and grader instead.
    Native Modal mounts use the S3 gateway with a named Secret and are read-only.
    """

    type: Literal["huggingface"] = "huggingface"
    bucket: str = Field(pattern=r"^[\w][\w.-]*/[\w][\w.-]*$")
    """Hugging Face namespace/bucket."""
    prefix: str = Field(default="", pattern=r"^([^\x00]*/)?$")
    """Object-key prefix, empty or ending in '/'."""
    modal_secret: str | None = Field(default=None, min_length=1)
    """Existing Modal Secret with Hugging Face-generated S3 credentials under
    AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, and AWS_REGION=us-east-1.
    Native Modal mounts use the S3 gateway, not HF_TOKEN, and are read-only:
    its upload checksum requirements cannot be configured through Modal."""


class HuggingFaceVolume(HostVolume):
    config: HuggingFaceVolumeConfig

    def _mount_command(self, path: Path) -> list[str]:
        if self.config.modal_secret is not None:
            raise ValueError(
                "modal_secret requires a native ModalConfig.mounts attachment"
            )
        command = ["hf-mount-fuse"]
        if self.config.read_only:
            command.append("--read-only")
        source = self.config.bucket
        if self.config.prefix:
            source += "/" + self.config.prefix.removesuffix("/")
        return [*command, "bucket", source, str(path)]
