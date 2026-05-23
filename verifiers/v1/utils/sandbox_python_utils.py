import shlex

SANDBOX_PYTHON_VERSION = "3.11"
SANDBOX_BIN_DIR = "/opt/verifiers/bin"
SANDBOX_PYTHON_ROOT = "/opt/verifiers/python"
SANDBOX_PYTHON_BIN_DIR = f"{SANDBOX_PYTHON_ROOT}/bin"
SANDBOX_PYTHON = f"{SANDBOX_PYTHON_BIN_DIR}/python3"
SANDBOX_UV = f"{SANDBOX_BIN_DIR}/uv"
SANDBOX_DEFAULT_PATH = (
    f"{SANDBOX_PYTHON_BIN_DIR}:"
    f"{SANDBOX_BIN_DIR}:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def uv_setup_command() -> str:
    return sandbox_runtime_script() + "vf_ensure_uv\n"


def python_runtime_setup_command() -> str:
    return sandbox_runtime_script() + "vf_ensure_python\n"


def python_package_install_command(package_args: str = "") -> str:
    command = python_runtime_setup_command()
    if package_args:
        command += f"vf_python_install {package_args}\n"
    return command


def python_package_list(value: object, field: str = "sandbox.packages") -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"{field} must be a list or string.")


def sandbox_runtime_script() -> str:
    return (
        "set -e\n"
        "export UV_NO_CONFIG=1\n"
        "export UV_NO_PROGRESS=1\n"
        "unset UV_INDEX_URL UV_EXTRA_INDEX_URL PIP_INDEX_URL PIP_EXTRA_INDEX_URL\n"
        f"VF_BIN_DIR={shlex.quote(SANDBOX_BIN_DIR)}\n"
        f"VF_PYTHON_ROOT={shlex.quote(SANDBOX_PYTHON_ROOT)}\n"
        f"VF_PYTHON={shlex.quote(SANDBOX_PYTHON)}\n"
        f"VF_UV={shlex.quote(SANDBOX_UV)}\n"
        "VF_PYTHON_INDEX_URL=https://pypi.org/simple\n"
        f"VF_PYTHON_VERSION={shlex.quote(SANDBOX_PYTHON_VERSION)}\n"
        'export PATH="$VF_PYTHON_ROOT/bin:$VF_BIN_DIR:$PATH"\n'
        'mkdir -p "$VF_BIN_DIR"\n'
        "vf_ensure_uv() {\n"
        'if [ ! -x "$VF_UV" ]; then\n'
        "  if command -v uv >/dev/null 2>&1; then\n"
        '    cp "$(command -v uv)" "$VF_UV"\n'
        "  else\n"
        "    if ! command -v curl >/dev/null 2>&1; then\n"
        "      if ! command -v apt-get >/dev/null 2>&1; then\n"
        "        echo 'curl or apt-get is required to install uv' >&2; exit 127\n"
        "      fi\n"
        "      apt-get -o Acquire::Retries=3 update && "
        "apt-get -o Acquire::Retries=3 install -y curl ca-certificates\n"
        "    fi\n"
        "    if ! curl -LsSf https://astral.sh/uv/install.sh -o /tmp/vf-uv-install.sh; then\n"
        "      if ! command -v apt-get >/dev/null 2>&1; then exit 1; fi\n"
        "      apt-get -o Acquire::Retries=3 update && "
        "apt-get -o Acquire::Retries=3 install -y ca-certificates\n"
        "      curl -LsSf https://astral.sh/uv/install.sh -o /tmp/vf-uv-install.sh\n"
        "    fi\n"
        '    env UV_INSTALL_DIR="$VF_BIN_DIR" sh /tmp/vf-uv-install.sh\n'
        "    rm -f /tmp/vf-uv-install.sh\n"
        "  fi\n"
        "fi\n"
        "}\n"
        "vf_ensure_python() {\n"
        "vf_ensure_uv\n"
        'if [ ! -x "$VF_PYTHON" ]; then\n'
        '  "$VF_UV" venv --python "$VF_PYTHON_VERSION" "$VF_PYTHON_ROOT"\n'
        "fi\n"
        'if [ ! -x "$VF_PYTHON" ] && [ -x "$VF_PYTHON_ROOT/bin/python" ]; then\n'
        '  ln -sfn "$VF_PYTHON_ROOT/bin/python" "$VF_PYTHON"\n'
        "fi\n"
        'if [ ! -x "$VF_PYTHON_ROOT/bin/python" ] && [ -x "$VF_PYTHON" ]; then\n'
        '  ln -sfn "$VF_PYTHON" "$VF_PYTHON_ROOT/bin/python"\n'
        "fi\n"
        "}\n"
        "vf_python_install() {\n"
        "vf_ensure_python\n"
        '  "$VF_UV" pip install --python "$VF_PYTHON" '
        '--index-url "$VF_PYTHON_INDEX_URL" "$@"\n'
        "}\n"
    )


def python_runtime_command(script_path: str, *args: str) -> list[str]:
    return [SANDBOX_PYTHON, script_path, *args]
