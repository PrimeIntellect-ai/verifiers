"""Hugging Face Storage Buckets through the official hf-mount client."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from verifiers.v1.volumes.base import BaseVolumeConfig, HostVolume


class HuggingFaceVolumeConfig(BaseVolumeConfig):
    """An existing Storage Bucket, not a model or dataset repository.

    Requires `hf-mount-fuse` and FUSE on the Linux host; authentication uses
    `HF_TOKEN` in the host environment. Writes use the client's streaming mode:
    write sequentially and close/fsync before grading. Existing files can be
    replaced with O_TRUNC; random writes are unsupported. Separate mounts may see stale data and have no
    shared file locks; share one attachment with the agent and grader instead.
    """

    type: Literal["huggingface"] = "huggingface"
    bucket: str = Field(pattern=r"^[\w][\w.-]*/[\w][\w.-]*$")
    """Hugging Face namespace/bucket."""
    prefix: str = Field(default="", pattern=r"^([^\x00]*/)?$")
    """Object-key prefix, empty or ending in '/'."""


class HuggingFaceVolume(HostVolume):
    config: HuggingFaceVolumeConfig

    def _mount_command(self, path: Path) -> list[str]:
        command = ["hf-mount-fuse"]
        if self.config.read_only:
            command.append("--read-only")
        source = self.config.bucket
        if self.config.prefix:
            source += "/" + self.config.prefix.removesuffix("/")
        return [*command, "bucket", source, str(path)]
