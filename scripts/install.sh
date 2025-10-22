#!/usr/bin/env bash
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

REPO_ID="verifiers"

has_ssh_access() {
    # Probe SSH auth to GitHub without prompting; treat any nonzero as "no ssh"
    # We try a quick ls-remote to avoid cloning on failure.
    # Disable -e for the probe so the script doesn't exit on a failed test.
    set +e
    timeout 5s git ls-remote --heads git@github.com:PrimeIntellect-ai/${REPO_ID}.git >/dev/null 2>&1
    rc=$?
    set -e
    return $rc
}

ensure_known_hosts() {
    # Make sure ~/.ssh exists with the right perms, then add GitHub host key.
    mkdir -p "${HOME}/.ssh"
    chmod 700 "${HOME}/.ssh"
    # Use -H to hash hostnames; merge uniquely to avoid dupes.
    if command -v ssh-keyscan >/dev/null 2>&1; then
    ssh-keyscan -H github.com 2>/dev/null | sort -u \
        | tee -a "${HOME}/.ssh/known_hosts" >/dev/null
    chmod 600 "${HOME}/.ssh/known_hosts"
    fi
}

main() {
    # Ensure sudo exists
    if ! command -v sudo &>/dev/null; then
        apt update
        apt install -y sudo
    fi

    log_info "Installing base packages..."
    sudo apt update && sudo apt install -y build-essential openssh-client curl git tmux htop nvtop

    log_info "Configuring SSH known_hosts for GitHub..."
    ensure_known_hosts

    log_info "Determining best way to clone (SSH vs HTTPS)..."
    if has_ssh_access; then
        log_info "SSH access to GitHub works. Cloning via SSH."
        git clone git@github.com:PrimeIntellect-ai/${REPO_ID}.git
    else
        log_warn "SSH auth to GitHub not available. Cloning via HTTPS."
        git clone https://github.com/PrimeIntellect-ai/${REPO_ID}.git
    fi

    log_info "Entering project directory..."
    cd ${REPO_ID}

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    log_info "Sourcing uv environment..."
    if ! command -v uv &> /dev/null; then
        source $HOME/.local/bin/env
    fi

    log_info "Syncing virtual environment..."
    uv sync

    log_info "Installing pre-commit hooks..."
    uv run pre-commit install

    log_info "Installation completed!"
}

main