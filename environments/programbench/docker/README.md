# programbench-toolchain

Docker image providing the multi-language toolchain for ProgramBench evaluation rollouts in prime-sandboxes.

## Contents

| Component | Version |
|-----------|---------|
| Base OS | Ubuntu 22.04 |
| Rust | stable (via rustup) |
| Go | 1.22.5 |
| C/C++ | gcc, g++, clang (Ubuntu 22.04 defaults, ~11/14) |
| Build tools | cmake, make |
| Analysis | binutils (strings, nm, objdump), file |
| Python | python3 + pip, pytest, pytest-xdist, junitparser |
| General | git, curl, wget, tar, bash |

The cargo registry is pre-warmed with: `clap`, `serde`, `serde_json`, `anyhow`, `tokio`, `regex`, `thiserror` (both debug and release profiles). This avoids 500 MB–2 GB of network downloads per rollout.

## Build and push

```bash
# One-time setup (only needed once per Docker host)
docker login
docker buildx create --use

# Build and push
./build.sh

# Build with a specific tag
TAG=v1.0 ./build.sh
```

## Expected image size

| Layer | Approx. size |
|-------|-------------|
| Ubuntu 22.04 base + apt packages | ~350 MB |
| Python + pytest deps | ~50 MB |
| Go 1.22 toolchain | ~500 MB |
| Rust toolchain (rustup + stable) | ~800 MB |
| Pre-warmed cargo registry | ~700 MB |
| **Total (compressed)** | **~2–2.5 GB** |
| **Total (uncompressed on disk)** | **~4–5 GB** |

The pre-warmed cargo layer adds ~700 MB compressed but eliminates multi-gigabyte downloads at eval time, which is the right trade-off for a long-lived sandbox image.
