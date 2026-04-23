"""Small helpers for install scripts that verify downloaded artifacts."""

from __future__ import annotations

import shlex
import textwrap

NVM_INSTALL_VERSION = "0.40.2"
NVM_INSTALL_SHA256 = "a909fdd01765379ebc5983674adafb8bc9de6d928bfa188761309d4a0c36be0f"
NVM_INSTALL_URL = (
    f"https://raw.githubusercontent.com/nvm-sh/nvm/v{NVM_INSTALL_VERSION}/install.sh"
)


def build_verified_nvm_install_command() -> str:
    """Return a shell block that hashes the pinned nvm installer before running."""
    return f"""\
NVM_INSTALL_SCRIPT="$(mktemp)"
curl -fsSL {shlex.quote(NVM_INSTALL_URL)} -o "$NVM_INSTALL_SCRIPT"
echo "{NVM_INSTALL_SHA256}  $NVM_INSTALL_SCRIPT" | sha256sum -c -
bash "$NVM_INSTALL_SCRIPT"
rm -f "$NVM_INSTALL_SCRIPT"
"""


def indent_shell_block(script: str, prefix: str = "  ") -> str:
    """Indent generated shell blocks before embedding them in branches."""
    return textwrap.indent(script.rstrip(), prefix)


def build_verified_npm_install_command(
    package_name: str,
    package_version: str,
    package_sha256: str,
    variable_prefix: str,
) -> str:
    """Return a shell block that hashes an npm tarball before installing it."""
    package = package_name
    if package_version:
        package = f"{package_name}@{package_version}"
    package_var = f"{variable_prefix}_PACKAGE"
    tarball_dir_var = f"{variable_prefix}_TARBALL_DIR"
    tarball_var = f"{variable_prefix}_TARBALL"

    return f"""\
{package_var}={shlex.quote(package)}
{tarball_dir_var}="$(mktemp -d)"
trap 'rm -rf "${{{tarball_dir_var}}}"' EXIT
npm pack "${{{package_var}}}" --pack-destination "${{{tarball_dir_var}}}" >/dev/null
{tarball_var}="$(find "${{{tarball_dir_var}}}" -maxdepth 1 -type f -name '*.tgz' -print -quit)"
if [ -z "${{{tarball_var}}}" ]; then
  echo "npm pack did not create a tarball for ${{{package_var}}}" >&2
  exit 1
fi
echo "{package_sha256}  ${{{tarball_var}}}" | sha256sum -c -
npm install -g "${{{tarball_var}}}"
"""
