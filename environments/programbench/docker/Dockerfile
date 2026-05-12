FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/root/.cargo \
    RUSTUP_HOME=/root/.rustup \
    GOPATH=/root/go \
    GOROOT=/usr/local/go \
    GO_VERSION=1.22.5 \
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

# Go 1.22.x
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
RUN mkdir -p /tmp/warmup/src && \
    cat > /tmp/warmup/Cargo.toml <<'EOF'
[package]
name = "warmup"
version = "0.1.0"
edition = "2021"

[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
tokio = { version = "1", features = ["full"] }
regex = "1"
thiserror = "1"

[[bin]]
name = "warmup"
path = "src/main.rs"
EOF
    cat > /tmp/warmup/src/main.rs <<'EOF'
use anyhow::Result;
use clap::Parser;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, default_value = "world")]
    name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Greeting {
    message: String,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("invalid name: {0}")]
    InvalidName(String),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let re = Regex::new(r"^\w+$")?;
    if !re.is_match(&cli.name) {
        return Err(AppError::InvalidName(cli.name).into());
    }
    let greeting = Greeting { message: format!("Hello, {}!", cli.name) };
    println!("{}", json!(greeting));
    Ok(())
}
EOF
    cd /tmp/warmup && \
    /root/.cargo/bin/cargo build 2>&1 && \
    /root/.cargo/bin/cargo build --release 2>&1 && \
    cd / && rm -rf /tmp/warmup

# Workspace directory used by sandbox rollouts
RUN mkdir -p /workspace

WORKDIR /workspace
