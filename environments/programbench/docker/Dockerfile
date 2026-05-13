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

# Rust via rustup (paper spec: 1.92.0)
RUN curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh && \
    sh /tmp/rustup-init.sh -y --default-toolchain 1.92.0 --profile minimal --no-modify-path && \
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
# We install system-wide (no --target) to use pre-built binary wheels and avoid
# source compilation overhead. The framework's install script skips the download
# loop when /opt/mini-swe-agent/prefix/bin/mini already exists.
# site-packages is left as a stub dir; Python finds the modules via the system path.
ARG MINI_SWE_AGENT_VERSION=2.2.8
RUN pip3 install --no-cache-dir "mini-swe-agent==${MINI_SWE_AGENT_VERSION}" && \
    mkdir -p /opt/mini-swe-agent/prefix/bin \
             /opt/mini-swe-agent/prefix/site-packages \
             /opt/mini-swe-agent/uv-site-packages \
             /logs/agent \
             /mini-swe-agent && \
    command -v python3 > /opt/mini-swe-agent/prefix/python && \
    printf '#!/usr/bin/env sh\nexport PYTHONPATH=/opt/mini-swe-agent/prefix/site-packages:${PYTHONPATH:-}\nexec "$(cat /opt/mini-swe-agent/prefix/python)" -m minisweagent.run.mini "$@"\n' \
        > /opt/mini-swe-agent/prefix/bin/mini && \
    chmod +x /opt/mini-swe-agent/prefix/bin/mini

# Workspace directory used by sandbox rollouts
RUN mkdir -p /workspace

WORKDIR /workspace
