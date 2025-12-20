"""
Cloudflare tunnel utilities for exposing local servers to remote sandboxes.

This module provides utilities for creating and managing Cloudflare Quick Tunnels,
which allow code running in remote sandboxes to make HTTP requests back to the
local machine without requiring manual network configuration.
"""

import asyncio
import logging
import platform
import shutil
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)


def ensure_cloudflared_installed() -> str:
    """
    Install cloudflared if not already installed.

    Returns:
        Path to the cloudflared binary.

    Raises:
        RuntimeError: If installation fails or platform is unsupported.
    """
    cloudflared_path = shutil.which("cloudflared")
    if cloudflared_path:
        return cloudflared_path

    logger.info("Installing cloudflared...")
    system = platform.system()

    if system == "Darwin":  # macOS
        result = subprocess.run(
            ["brew", "install", "cloudflare/cloudflare/cloudflared"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to install cloudflared via Homebrew: {result.stderr}"
            )
        cloudflared_path = shutil.which("cloudflared")
        if not cloudflared_path:
            raise RuntimeError("cloudflared installed but not found in PATH")
        return cloudflared_path
    elif system == "Linux":
        install_script = (
            "curl -L --output cloudflared.deb "
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb "
            "&& sudo dpkg -i cloudflared.deb && rm cloudflared.deb"
        )
        result = subprocess.run(
            ["bash", "-c", install_script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to install cloudflared: {result.stderr}")
        cloudflared_path = shutil.which("cloudflared")
        if not cloudflared_path:
            raise RuntimeError("cloudflared installed but not found in PATH")
        return cloudflared_path
    else:
        raise RuntimeError(
            f"Unsupported platform: {system}. "
            "Please install cloudflared manually: "
            "https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
        )


def extract_tunnel_url_from_line(line: str) -> str | None:
    """
    Extract tunnel URL from a line of cloudflared output.

    Args:
        line: A line of stderr output from cloudflared.

    Returns:
        The tunnel URL if found, None otherwise.
    """
    if ".trycloudflare.com" not in line:
        return None

    start_idx = line.find("https://")
    if start_idx == -1:
        return None

    url_end = start_idx + 8  # Skip "https://"
    while url_end < len(line) and not line[url_end].isspace():
        url_end += 1

    url = line[start_idx:url_end].rstrip("/")
    if ".trycloudflare.com" in url:
        return url
    return None


def start_cloudflared_tunnel(
    port: int, max_wait_seconds: int = 30
) -> tuple[str, subprocess.Popen]:
    """
    Start a cloudflared tunnel and return the URL and process.

    Args:
        port: Local port to tunnel to.
        max_wait_seconds: Maximum time to wait for tunnel URL.

    Returns:
        Tuple of (tunnel_url, tunnel_process).

    Raises:
        RuntimeError: If tunnel fails to start or URL not obtained in time.
    """
    cloudflared_path = ensure_cloudflared_installed()

    tunnel_process = subprocess.Popen(
        [
            cloudflared_path,
            "tunnel",
            "--url",
            f"http://localhost:{port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stderr_lines: list[str] = []
    check_interval = 0.5
    max_iterations = int(max_wait_seconds / check_interval)

    for _ in range(max_iterations):
        if tunnel_process.poll() is not None:
            if tunnel_process.stderr:
                remaining = tunnel_process.stderr.read()
                stderr_lines.append(remaining)
            error_output = "".join(stderr_lines)
            raise RuntimeError(f"cloudflared tunnel failed to start: {error_output}")

        if tunnel_process.stderr:
            line = tunnel_process.stderr.readline()
            if line:
                stderr_lines.append(line)
                url = extract_tunnel_url_from_line(line)
                if url:
                    logger.info(f"Cloudflare tunnel started: {url}")
                    return url, tunnel_process

        time.sleep(check_interval)

    raise RuntimeError(
        f"Failed to get tunnel URL from cloudflared after {max_wait_seconds} seconds."
    )


class TunnelPool:
    """
    Manages a pool of Cloudflare tunnels with automatic scaling.

    Creates tunnels on demand and distributes rollouts across them using
    round-robin selection. Automatically scales up when more rollouts are active.

    Args:
        port: Local port to tunnel to.
        rollouts_per_tunnel: Maximum rollouts per tunnel before scaling up.
        max_wait_seconds: Maximum time to wait for each tunnel to start.
    """

    def __init__(
        self,
        port: int,
        rollouts_per_tunnel: int = 50,
        max_wait_seconds: int = 30,
    ):
        self.port = port
        self.rollouts_per_tunnel = rollouts_per_tunnel
        self.max_wait_seconds = max_wait_seconds
        self._tunnels: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._round_robin_index = 0

    async def get_tunnel_url(self, active_rollout_count: int) -> str:
        """
        Get a tunnel URL, creating new tunnels as needed.

        Args:
            active_rollout_count: Current number of active rollouts.

        Returns:
            The tunnel URL to use for this rollout.

        Raises:
            RuntimeError: If tunnel creation fails.
        """
        async with self._lock:
            required_tunnels = max(
                1,
                (active_rollout_count + self.rollouts_per_tunnel - 1)
                // self.rollouts_per_tunnel,
            )

            while len(self._tunnels) < required_tunnels:
                try:
                    url, process = start_cloudflared_tunnel(
                        self.port, self.max_wait_seconds
                    )
                    self._tunnels.append(
                        {
                            "url": url,
                            "process": process,
                            "active_rollouts": 0,
                        }
                    )
                    logger.debug(
                        f"Created tunnel {len(self._tunnels)}/{required_tunnels}: {url}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create tunnel: {e}")
                    raise

            tunnel = self._tunnels[self._round_robin_index % len(self._tunnels)]
            self._round_robin_index += 1
            tunnel["active_rollouts"] += 1
            return tunnel["url"]

    async def release_tunnel(self, tunnel_url: str) -> None:
        """
        Release a tunnel URL, decrementing its active rollout count.

        Args:
            tunnel_url: The tunnel URL to release.
        """
        async with self._lock:
            for tunnel in self._tunnels:
                if tunnel["url"] == tunnel_url:
                    tunnel["active_rollouts"] = max(0, tunnel["active_rollouts"] - 1)
                    break

    def teardown(self) -> None:
        """
        Stop all tunnel processes.

        Safe to call from sync context (e.g., signal handlers).
        """
        for tunnel in self._tunnels:
            process = tunnel.get("process")
            if process:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception as e:
                    logger.warning(f"Error stopping tunnel: {e}")
                    try:
                        process.kill()
                    except Exception:
                        pass
        self._tunnels.clear()
        logger.debug("All tunnels stopped")
