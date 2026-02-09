import importlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

ENVIRONMENTS_HUB_URL = "https://api.primeintellect.ai/api/v1/environmentshub"


def _uv_pip_cmd(subcommand: str, *args: str) -> list[str]:
    """Run uv pip against the active Python interpreter for this process."""
    return ["uv", "pip", subcommand, "--python", sys.executable, *args]


def normalize_package_name(name: str) -> str:
    """Normalize package name according to Python packaging standards."""
    return name.replace("-", "_").lower()


def parse_env_id(env_id: str) -> tuple[str, str, Optional[str]]:
    """Parse environment ID into (owner, name, version).

    Args:
        env_id: Environment ID like 'owner/name' or 'owner/name@version'

    Returns:
        Tuple of (owner, name, version). Version is None if not specified.

    Raises:
        ValueError: If format is invalid
    """
    version = None
    if "@" in env_id:
        env_id, version = env_id.rsplit("@", 1)

    parts = env_id.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid environment ID: '{env_id}'. Expected format: 'owner/name' or 'owner/name@version'"
        )

    return parts[0], parts[1], version


def is_hub_env(env_id: str) -> bool:
    """Check if env_id refers to a Hub environment (has owner/ prefix)."""
    return "/" in env_id and not env_id.startswith("./") and not env_id.startswith("/")


def is_installed(env_name: str, version: Optional[str] = None) -> bool:
    """Check if an environment package is installed.

    Args:
        env_name: Environment name (without owner prefix)
        version: Optional version to check for

    Returns:
        True if installed (and version matches if specified)
    """
    try:
        pkg_name = normalize_package_name(env_name)
        result = subprocess.run(
            _uv_pip_cmd("show", pkg_name),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return False

        if version and version != "latest":
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    installed_version = line.split(":", 1)[1].strip()
                    return installed_version == version
            return False

        return True
    except Exception:
        return False


def check_hub_env_installed(env_id: str) -> bool:
    """Check if a Hub environment is installed.

    For Hub environments (owner/name format), checks if the package is installed.
    For local environments, returns True (assumes available via env_dir).

    Args:
        env_id: Environment ID (local name or 'owner/name' or 'owner/name@version')

    Returns:
        True if environment is available (or is local)
    """
    if not is_hub_env(env_id):
        return True

    _, name, version = parse_env_id(env_id)
    return is_installed(name, version)


def _is_valid_url(url: str) -> bool:
    """Check if a string is a valid HTTP(S) URL."""
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def install_from_hub(env_id: str) -> bool:
    """Install an environment from the Environments Hub.

    Args:
        env_id: Environment ID like 'owner/name' or 'owner/name@version'

    Returns:
        True if installation succeeded
    """
    owner, name, version = parse_env_id(env_id)
    version = version or "latest"

    logger.info(f"Fetching environment details for {owner}/{name}@{version}...")

    try:
        url = f"{ENVIRONMENTS_HUB_URL}/{owner}/{name}/@{version}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        details = data.get("data", data)
    except requests.RequestException as e:
        logger.error(f"Failed to fetch environment details: {e}")
        return False
    except ValueError as e:
        logger.error(f"Invalid response from Environments Hub: {e}")
        return False

    simple_index_url = details.get("simple_index_url")
    wheel_url = details.get("wheel_url")
    if wheel_url and not _is_valid_url(wheel_url):
        wheel_url = None

    if not simple_index_url and not wheel_url:
        if details.get("visibility") == "PRIVATE":
            logger.error(
                f"Cannot install private environment '{env_id}'. "
                "Use 'prime env pull' to download and install locally."
            )
        else:
            logger.error(f"No installation method available for '{env_id}'")
        return False

    pkg_name = normalize_package_name(name)

    cmd: list[str]
    if simple_index_url:
        if version and version != "latest":
            pkg_spec = f"{pkg_name}=={version}"
        else:
            pkg_spec = pkg_name
        cmd = [
            *_uv_pip_cmd("install"),
            "--upgrade",
            pkg_spec,
            "--extra-index-url",
            simple_index_url,
        ]
    else:
        assert wheel_url is not None
        cmd = [*_uv_pip_cmd("install"), "--upgrade", wheel_url]

    logger.info(f"Installing {env_id}...")
    logger.debug(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Installation failed: {result.stderr}")
        return False

    logger.info(f"Successfully installed {env_id}")
    importlib.invalidate_caches()
    return True


def install_from_local(env_name: str, env_dir: str = "./environments") -> bool:
    """Install an environment from a local directory.

    Args:
        env_name: Environment name
        env_dir: Path to environments directory

    Returns:
        True if installation succeeded
    """
    env_folder = normalize_package_name(env_name)
    env_path = Path(env_dir) / env_folder

    if not env_path.exists():
        logger.error(f"Local environment not found: {env_path}")
        return False

    logger.info(f"Installing {env_name} from {env_path}...")
    result = subprocess.run(
        [*_uv_pip_cmd("install"), "-e", str(env_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Installation failed: {result.stderr}")
        return False

    logger.info(f"Successfully installed {env_name}")
    importlib.invalidate_caches()
    return True


def install_from_repo(env_name: str, branch: str = "main") -> bool:
    """Install an environment from the verifiers GitHub repo.

    Args:
        env_name: Environment name
        branch: Git branch to install from

    Returns:
        True if installation succeeded
    """
    env_folder = normalize_package_name(env_name)
    pkg_name = env_folder.replace("_", "-")

    url = f"git+https://github.com/PrimeIntellect-ai/verifiers.git@{branch}#subdirectory=environments/{env_folder}"

    logger.info(f"Installing {env_name} from verifiers repo ({branch})...")
    result = subprocess.run(
        [*_uv_pip_cmd("install"), f"{pkg_name} @ {url}"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Installation failed: {result.stderr}")
        return False

    logger.info(f"Successfully installed {env_name}")
    importlib.invalidate_caches()
    return True
