from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("verifiers")
except PackageNotFoundError:  # source tree without install metadata
    __version__ = "0.0.0+unknown"
