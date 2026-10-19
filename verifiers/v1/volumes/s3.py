"""Amazon S3 attachments through the official Mountpoint client."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from verifiers.v1.volumes.base import BaseVolumeConfig, HostVolume


class S3VolumeConfig(BaseVolumeConfig):
    """Requires `mount-s3` and FUSE on Linux, on the container engine's host.

    Authentication uses Mountpoint's AWS credential chain (environment, profiles,
    or IAM role). Writable mounts support creating files with sequential writes;
    overwrite, delete, and random writes are not enabled. Rename is limited to
    S3 Express One Zone. Call fsync and check its result before closing files for
    grading or teardown; an unmount is not a commit.
    """

    type: Literal["s3"] = "s3"
    bucket: str = Field(pattern=r"^[a-z0-9][a-z0-9.-]*$")
    prefix: str = Field(default="", pattern=r"^([^\x00]*/)?$")
    """Object-key prefix, empty or ending in '/'."""
    region: str | None = None
    profile: str | None = None
    anonymous: bool = False
    """Disable request signing for public buckets."""


class S3Volume(HostVolume):
    config: S3VolumeConfig

    def _mount_command(self, path: Path) -> list[str]:
        command = [
            "mount-s3",
            self.config.bucket,
            str(path),
            "--foreground",
            "--allow-other",
            "--auto-unmount",
        ]
        for flag, value in (
            ("--prefix", self.config.prefix),
            ("--region", self.config.region),
            ("--profile", self.config.profile),
        ):
            if value:
                command.extend([flag, value])
        if self.config.read_only:
            command.append("--read-only")
        if self.config.anonymous:
            command.append("--no-sign-request")
        return command
