"""The verifiers package root.

The v1 stack lives in `verifiers.v1` (`import verifiers.v1 as vf`). The
classic v0 stack (`verifiers.legacy`, which also answered at its historical
top-level paths — `verifiers.envs`, `verifiers.types`, ...) has been removed.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("verifiers")
except PackageNotFoundError:  # source tree without install metadata
    __version__ = "0.0.0+unknown"
