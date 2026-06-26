"""Resolve and install environment packages from the Prime Environments Hub."""

import hashlib
import importlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from prime_sandboxes import APIClient

logger = logging.getLogger(__name__)


def normalize_package_name(name: str) -> str:
    """Return the importable spelling used by environment packages."""
    return name.replace("-", "_").lower()


def parse_env_id(env_id: str) -> tuple[str, str, str | None]:
    """Parse ``owner/name[@version]`` into its three components."""
    version = None
    if "@" in env_id:
        env_id, version = env_id.rsplit("@", 1)

    parts = env_id.split("/")
    valid_parts = len(parts) == 2 and all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in parts
    )
    valid_version = version is None or bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", version)
    )
    if not valid_parts or not valid_version:
        raise ValueError(
            f"Invalid environment ID: {env_id!r}. Expected 'owner/name' or "
            "'owner/name@version'."
        )
    return parts[0], parts[1], version


def is_hub_env(env_id: str) -> bool:
    """Whether an id is an owner-qualified Hub reference."""
    return "/" in env_id and not env_id.startswith(("./", "/"))


def fetch_hub_environment(
    env_id: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Fetch a public or private Hub environment with the active Prime context."""
    from verifiers.utils.client_utils import load_prime_config

    owner, name, version = parse_env_id(env_id)
    prime_config = load_prime_config()
    client = APIClient(
        api_key=api_key or prime_config.get("api_key"), require_auth=False
    )
    client.base_url = (
        (base_url or prime_config["base_url"]).rstrip("/").removesuffix("/api/v1")
    )
    response = client.request(
        "GET", f"/environmentshub/{owner}/{name}/@{version or 'latest'}"
    )
    details = response.get("data", response)
    if not isinstance(details, dict):
        raise ValueError(f"Invalid Environments Hub response for {env_id!r}")
    return details


def environment_package_url(details: dict[str, Any]) -> str | None:
    """Return the downloadable source package URL from Hub details."""
    for key in ("tracked_package_url", "package_url"):
        value = details.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract a source archive without links, absolute paths, or traversal."""
    destination = destination.resolve()
    for member in tar.getmembers():
        path = Path(member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"Refusing to extract archive link: {member.name}")
        if not member.isfile() and not member.isdir():
            raise ValueError(
                f"Refusing to extract special archive entry: {member.name}"
            )
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Refusing to extract unsafe path: {member.name}")
        if not (destination / path).resolve().is_relative_to(destination):
            raise ValueError(f"Archive path escapes destination: {member.name}")
    tar.extractall(destination)


def download_environment_source(
    details: dict[str, Any],
    destination: Path,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Path:
    """Download and safely extract a Hub source package into ``destination``."""
    package_url = environment_package_url(details)
    parsed = urlparse(package_url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Environment has no valid downloadable source package")
    assert package_url is not None

    from verifiers.utils.client_utils import load_prime_config

    prime_config = load_prime_config()
    key = api_key or prime_config.get("api_key")
    api_host = urlparse(base_url or prime_config["base_url"]).netloc
    headers = (
        {"Authorization": f"Bearer {key}"} if key and parsed.netloc == api_host else {}
    )
    destination = destination.expanduser().absolute()
    if destination.is_symlink() or destination == Path(destination.anchor):
        raise ValueError(f"Unsafe extraction destination: {destination}")
    with tempfile.TemporaryDirectory(prefix="vf-env-source-") as directory:
        archive = Path(directory) / "environment.tar.gz"
        extracted = Path(directory) / "extracted"
        extracted.mkdir()
        with httpx.stream(
            "GET", package_url, headers=headers, timeout=60.0, follow_redirects=True
        ) as response:
            response.raise_for_status()
            with archive.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=8192):
                    handle.write(chunk)
        with tarfile.open(archive, "r:gz") as tar:
            safe_extract(tar, extracted)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extracted, destination, dirs_exist_ok=True)
    return destination


def build_environment_wheel(source: Path, cache_dir: Path) -> Path:
    """Build a source environment and atomically cache its wheel."""
    with tempfile.TemporaryDirectory(prefix="vf-env-wheel-") as directory:
        dist = Path(directory) / "dist"
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(dist)],
            cwd=source,
            check=True,
        )
        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"Expected one built wheel for {source}, found {len(wheels)}"
            )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / wheels[0].name
        temporary = cache_dir / f".{wheels[0].name}.{os.getpid()}"
        shutil.copy2(wheels[0], temporary)
        temporary.replace(cached)
    return cached


def install_from_hub(
    env_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    python_executable: str | Path | None = None,
    prerelease: bool = False,
) -> str:
    """Install a public wheel/index or private source package and return its module."""
    owner, name, version = parse_env_id(env_id)
    details = fetch_hub_environment(env_id, api_key=api_key, base_url=base_url)
    simple_index_url = details.get("simple_index_url")
    wheel_url = details.get("wheel_url")
    url_dependencies = details.get("url_dependencies") or []
    package = normalize_package_name(name)
    pinned_version = version if version not in (None, "latest") else None
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python_executable or sys.executable),
        "-P",
        package,
    ]

    if isinstance(simple_index_url, str) and simple_index_url:
        command.append(f"{package}=={pinned_version}" if pinned_version else package)
        command.extend(str(dependency) for dependency in url_dependencies)
        command.extend(
            [
                "--extra-index-url",
                simple_index_url,
                "--exclude-newer-package",
                f"{package}=false",
            ]
        )
    elif isinstance(wheel_url, str) and wheel_url:
        parsed = urlparse(wheel_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid wheel URL for {env_id!r}: {wheel_url}")
        command.append(wheel_url)
        command.extend(str(dependency) for dependency in url_dependencies)
    else:
        version_data = details.get("latest_version")
        content_hash = (
            version_data.get("content_hash")
            if isinstance(version_data, dict)
            else details.get("content_hash")
        )
        valid_hash = (
            isinstance(content_hash, str)
            and len(content_hash) == 64
            and all(
                character in "0123456789abcdef" for character in content_hash.lower()
            )
        )
        cache_root = Path.home() / ".cache" / "verifiers" / "hub" / owner / name
        cache_dir = cache_root / str(content_hash) if valid_hash else None
        wheels = list(cache_dir.glob("*.whl")) if cache_dir else []
        if wheels:
            wheel = wheels[0]
        else:
            with tempfile.TemporaryDirectory(prefix="vf-env-build-") as directory:
                source = download_environment_source(
                    details,
                    Path(directory) / "source",
                    api_key=api_key,
                    base_url=base_url,
                )
                if cache_dir is None:
                    digest = hashlib.sha256()
                    for path in sorted(
                        item for item in source.rglob("*") if item.is_file()
                    ):
                        digest.update(path.relative_to(source).as_posix().encode())
                        digest.update(b"\0")
                        digest.update(path.read_bytes())
                    cache_dir = cache_root / digest.hexdigest()
                cached = list(cache_dir.glob("*.whl"))
                wheel = (
                    cached[0] if cached else build_environment_wheel(source, cache_dir)
                )
        command.append(str(wheel))

    if prerelease:
        command.append("--prerelease=allow")
    logger.info("installing %s", env_id)
    subprocess.run(command, check=True)
    importlib.invalidate_caches()
    return package
