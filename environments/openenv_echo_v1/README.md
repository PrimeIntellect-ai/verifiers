# openenv-echo-v1

OpenEnv's official Echo image as a minimal v1 example. The reusable image lifecycle and
JSON-RPC-to-MCP bridge live in `verifiers`; this package only pins the image, prompt, and
resources.

## Develop

1. From the Verifiers checkout root, add both the current framework and this example to the
   ephemeral `uv run` environment.
2. Run it with any MCP-capable Verifiers harness and a container runtime:

```bash
uv run --with-editable . --with-editable environments/openenv_echo_v1 \
  eval openenv-echo-v1 -n 1 --harness.runtime.type docker
uv run --with-editable . --with-editable environments/openenv_echo_v1 \
  eval openenv-echo-v1 -n 1 --harness.runtime.type docker --harness.id rlm
uv run --with-editable . --with-editable environments/openenv_echo_v1 \
  eval openenv-echo-v1 -n 1 --harness.runtime.type docker --harness.id codex \
  -m nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B
```

## Layout

- `openenv_echo_v1/taskset.py` — a thin config over `vf`'s reusable `OpenEnvTaskset`.

Echo's production MCP contract is unscored, so the reward is neutral while the tool call and
result remain in the Verifiers trace.
