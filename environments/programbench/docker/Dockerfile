FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    GOPATH=/root/go \
    GOROOT=/usr/local/go \
    GO_VERSION=1.21.0 \
    PATH="/root/.cargo/bin:/usr/local/go/bin:/root/go/bin:${PATH}"

# System packages: C/C++ toolchain, analysis tools, Python, general utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        clang \
        cmake \
        make \
        binutils \
        file \
        python3 \
        python3-pip \
        python3-dev \
        tmux \
        git \
        curl \
        wget \
        tar \
        bash \
        ca-certificates \
        pkg-config \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Python tooling
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-xdist \
    junitparser

# Go (paper spec: 1.21.0)
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tar.gz && \
    tar -C /usr/local -xzf /tmp/go.tar.gz && \
    rm /tmp/go.tar.gz

# Rust via rustup (stable, no default toolchain prompt)
RUN curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh && \
    sh /tmp/rustup-init.sh -y --default-toolchain stable --profile minimal --no-modify-path && \
    rm /tmp/rustup-init.sh && \
    /root/.cargo/bin/rustup component add rustfmt clippy

# Pre-warm cargo registry with common ProgramBench dependencies.
# We build both debug and release profiles so that all dep artifacts are cached.
# The /tmp/warmup source tree is deleted afterwards; ~/.cargo/registry persists.
COPY warmup/ /tmp/warmup/
RUN cd /tmp/warmup && \
    /root/.cargo/bin/cargo build 2>&1 && \
    /root/.cargo/bin/cargo build --release 2>&1 && \
    cd / && rm -rf /tmp/warmup

# Pre-install mini-swe-agent so the harness can run with network_access=False.
# Mirrors the directory layout expected by build_mini_swe_agent_install_script:
#   /opt/mini-swe-agent/prefix/bin/mini   — entry-point wrapper
#   /opt/mini-swe-agent/prefix/python     — path to the python interpreter
#   /opt/mini-swe-agent/prefix/site-packages/  — installed wheel
# The framework's install script checks for /opt/mini-swe-agent/prefix/bin/mini
# and skips the download when it exists.
ARG MINI_SWE_AGENT_VERSION=2.2.8
ARG MINI_SWE_AGENT_SHA256=694df4de1337e665e3cd82e99f93374f573bf52b8e7c362ac5d8045ad9f7c37c
RUN set -e && \
    WHEEL_FILE="mini_swe_agent-${MINI_SWE_AGENT_VERSION}-py3-none-any.whl" && \
    WHEEL_URL="https://files.pythonhosted.org/packages/py3/m/mini-swe-agent/${WHEEL_FILE}" && \
    mkdir -p /opt/mini-swe-agent/prefix/bin \
             /opt/mini-swe-agent/prefix/site-packages \
             /opt/mini-swe-agent/uv-site-packages \
             /logs/agent \
             /mini-swe-agent && \
    curl -fsSL "${WHEEL_URL}" -o "/tmp/${WHEEL_FILE}" && \
    echo "${MINI_SWE_AGENT_SHA256}  /tmp/${WHEEL_FILE}" | sha256sum -c - && \
    pip3 install --quiet --no-cache-dir --target /opt/mini-swe-agent/prefix/site-packages "/tmp/${WHEEL_FILE}" && \
    rm "/tmp/${WHEEL_FILE}" && \
    command -v python3 > /opt/mini-swe-agent/prefix/python && \
    printf '#!/usr/bin/env sh\nexport PYTHONPATH=/opt/mini-swe-agent/prefix/site-packages:${PYTHONPATH:-}\nexec "$(cat /opt/mini-swe-agent/prefix/python)" -m minisweagent.run.mini "$@"\n' \
        > /opt/mini-swe-agent/prefix/bin/mini && \
    chmod +x /opt/mini-swe-agent/prefix/bin/mini

# Workspace directory used by sandbox rollouts
RUN mkdir -p /workspace

WORKDIR /workspace
