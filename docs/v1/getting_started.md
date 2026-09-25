# Installation

verifiers v1 requires Linux or macOS. Parts of the stack depend on Unix-only functionality (POSIX file locking in the runtime rate limiters, ZMQ `ipc://` sockets in the env server), so `import verifiers.v1` fails on native Windows even though `pip install verifiers` succeeds. On Windows, run verifiers under [WSL](https://learn.microsoft.com/windows/wsl/install).

verifiers runs locally with `uv`. Install it, clone the repo, and sync dependencies:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync
```

You can now run tasksets directly, e.g. `uv run vf-eval <taskset-id>`, and scaffold new ones with `uv run vf-init <name>`.

## Skills

To equip your agent with the necessary knowledge, we highly recommend the skills in this repository's [`skills/`](https://github.com/PrimeIntellect-ai/verifiers/tree/main/skills) directory (alongside [`AGENTS.md`](https://github.com/PrimeIntellect-ai/verifiers/blob/main/AGENTS.md)). They are more comprehensive than these docs, which are meant for human consumption.
