import importlib
import io
import json
import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

ENVIRONMENTS_HUB_URL = "https://api.primeintellect.ai/api/v1/environmentshub"


def _uv_pip_cmd(subcommand: str, *args: str) -> list[str]:
    """Run uv pip against the active Python interpreter for this process."""
    return ["uv", "pip", subcommand, "--python", sys.executable, *args]


def load_prime_config() -> dict:
    """Read `~/.prime/config.json` (the prime CLI's credential store), or {} if absent."""
    try:
        config_file = Path.home() / ".prime" / "config.json"
        if config_file.exists():
            data = json.loads(config_file.read_text())
            if isinstance(data, dict):
                return data
            logger.warning("Invalid prime config: expected dict")
    except (RuntimeError, json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load prime config: {e}")
    return {}


def prime_auth_headers() -> dict[str, str]:
    """Resolve the prime API credentials the same way the eval client does: `PRIME_API_KEY`
    (or `~/.prime/config.json`) for the bearer token, `PRIME_TEAM_ID` (or the config) for
    the team header. The Hub's metadata endpoint 404s an unauthenticated request even for
    PUBLIC envs the caller can otherwise list, so every Hub request sends these."""
    prime_config = load_prime_config()
    headers: dict[str, str] = {}
    api_key = os.getenv("PRIME_API_KEY") or prime_config.get("api_key", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    team_id = os.getenv("PRIME_TEAM_ID") or prime_config.get("team_id")
    if team_id:
        headers["X-Prime-Team-ID"] = team_id
    return headers


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
        response = requests.get(url, headers=prime_auth_headers(), timeout=30)
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
    if wheel_url:
        try:
            parsed_wheel_url = urlparse(wheel_url)
            if (
                parsed_wheel_url.scheme not in ("http", "https")
                or not parsed_wheel_url.netloc
            ):
                wheel_url = None
        except Exception:
            wheel_url = None

    # Private envs publish no wheel/index, only a (presigned) source archive — download and
    # build it locally, the same source-pull-and-build path `prime env install` takes.
    if not simple_index_url and not wheel_url:
        package_url = details.get("package_url")
        if not package_url:
            logger.error(f"No installation method available for '{env_id}'")
            return False
        return _install_from_source_archive(env_id, package_url)

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


def _install_from_source_archive(env_id: str, package_url: str) -> bool:
    """Download a source archive (`package_url`) and build+install it with uv.

    Private Hub envs publish no wheel or simple index, only this source tarball. The
    metadata fetch that yields `package_url` is authenticated, but the URL itself is
    presigned, so the download needs no further credentials."""
    logger.info(f"Installing {env_id} from source archive...")
    try:
        resp = requests.get(package_url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download source archive: {e}")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        with tarfile.open(fileobj=io.BytesIO(resp.content)) as tar:
            try:
                tar.extractall(src, filter="data")  # py3.12+: refuse unsafe members
            except TypeError:
                tar.extractall(src)
        result = subprocess.run(
            [*_uv_pip_cmd("install"), "--upgrade", str(src)],
            capture_output=True,
            text=True,
        )

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
