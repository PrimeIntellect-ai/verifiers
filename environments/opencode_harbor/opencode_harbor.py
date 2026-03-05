import json
import logging
from pathlib import Path

from verifiers.envs.experimental.harbor_env import HarborEnv

logger = logging.getLogger("verifiers.envs.OpenCodeHarborEnv")

TERMINAL_BENCH_SAMPLE_TASKS = [
    "build-cython-ext",
    "chess-best-move",
    "configure-git-webserver",
    "fix-code-vulnerability",
    "log-summary-date-ranges",
    "polyglot-c-py",
    "qemu-alpine-ssh",
    "qemu-startup",
    "regex-log",
    "sqlite-with-gcov",
]

# make sure these tasks are present locally in `tasks/` directory
TERMINAL_BENCH_TASKS = [
    "adaptive-rejection-sampler",
    "bn-fit-modify",
    "break-filter-js-from-html",
    "build-cython-ext",
    "build-pmars",
    "build-pov-ray",
    "caffe-cifar-10",
    "cancel-async-tasks",
    "chess-best-move",
    "circuit-fibsqrt",
    "cobol-modernization",
    "code-from-image",
    "compile-compcert",
    "configure-git-webserver",
    "constraints-scheduling",
    "count-dataset-tokens",
    "crack-7z-hash",
    "custom-memory-heap-crash",
    "db-wal-recovery",
    "distribution-search",
    "dna-assembly",
    "dna-insert",
    "extract-elf",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "filter-js-from-html",
    "financial-document-processor",
    "fix-code-vulnerability",
    "fix-git",
    "fix-ocaml-gc",
    "gcode-to-text",
    "git-leak-recovery",
    "git-multibranch",
    "gpt2-codegolf",
    "headless-terminal",
    "hf-model-inference",
    "install-windows-3.11",
    "kv-store-grpc",
    "large-scale-text-editing",
    "largest-eigenval",
    "llm-inference-batching-scheduler",
    "log-summary-date-ranges",
    "mailman",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "merge-diff-arc-agi-task",
    "model-extraction-relu-logits",
    "modernize-scientific-stack",
    "mteb-leaderboard",
    "mteb-retrieve",
    "multi-source-data-merger",
    "nginx-request-logging",
    "openssl-selfsigned-cert",
    "overfull-hbox",
    "password-recovery",
    "path-tracing",
    "path-tracing-reverse",
    "polyglot-c-py",
    "polyglot-rust-c",
    "portfolio-optimization",
    "protein-assembly",
    "prove-plus-comm",
    "pypi-server",
    "pytorch-model-cli",
    "pytorch-model-recovery",
    "qemu-alpine-ssh",
    "qemu-startup",
    "query-optimize",
    "raman-fitting",
    "regex-chess",
    "regex-log",
    "reshard-c4-data",
    "rstan-to-pystan",
    "sam-cell-seg",
    "sanitize-git-repo",
    "schemelike-metacircular-eval",
    "sparql-university",
    "sqlite-db-truncate",
    "sqlite-with-gcov",
    "torch-pipeline-parallelism",
    "torch-tensor-parallelism",
    "train-fasttext",
    "tune-mjcf",
    "video-processing",
    "vulnerable-secret",
    "winning-avg-corewars",
    "write-compressor",
]

DATASETS = {
    "terminal-bench": TERMINAL_BENCH_TASKS,
    "terminal-bench-sample": TERMINAL_BENCH_SAMPLE_TASKS,
}


def _build_opencode_config(
    disabled_tools: list[str] | None = None,
    system_prompt_path: str | None = None,
) -> str:
    config: dict = {
        "${SCHEMA_DOLLAR}schema": "https://opencode.ai/config.json",
        "provider": {
            "intercepted": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Intercepted",
                "options": {
                    "baseURL": "$OPENAI_BASE_URL",
                    "apiKey": "intercepted",
                    "timeout": 600000,
                },
                "models": {
                    "model": {
                        "name": "Intercepted Model",
                        "modalities": {"input": ["text", "image"], "output": ["text"]},
                    }
                },
            }
        },
        "model": "intercepted/model",
    }

    # Add agent config if we have custom prompt or disabled tools
    if system_prompt_path or disabled_tools:
        build_config: dict = {}

        if system_prompt_path:
            build_config["prompt"] = "{file:" + system_prompt_path + "}"

        if disabled_tools:
            build_config["tools"] = {tool: False for tool in disabled_tools}

        config["agent"] = {"build": build_config}

    return json.dumps(config, indent=2)


def _build_run_command(
    agent_workdir: str,
    disabled_tools: list[str] | None = None,
    has_system_prompt: bool = False,
) -> str:
    # Path where we'll upload the system prompt in the sandbox
    system_prompt_sandbox_path = "/opencode/prompt.txt" if has_system_prompt else None
    config_json = _build_opencode_config(disabled_tools, system_prompt_sandbox_path)

    return f"""
set -e

apt-get update && apt-get install -y curl

curl -fsSL https://opencode.ai/install | bash
export PATH="$HOME/.opencode/bin:$PATH"

# Create opencode config directory
mkdir -p ~/.config/opencode

# Preserve JSON schema key literal in unquoted heredoc while still expanding
# OPENAI_BASE_URL.
SCHEMA_DOLLAR='$'

# Create opencode.json config
cat > ~/.config/opencode/opencode.json << EOFCONFIG
{config_json}
EOFCONFIG

mkdir -p /logs/agent

# Run OpenCode with task instruction
cd {agent_workdir}
opencode run "$(cat /task/instruction.md)" 2>&1 | tee /logs/agent/opencode.txt
"""


class OpenCodeHarborEnv(HarborEnv):
    def __init__(
        self,
        dataset_path: str | Path,
        tasks: list[str] | None = None,
        agent_workdir: str = "/app",
        docker_image: str = "python:3.11-slim",
        system_prompt_path: str | Path | None = None,
        disabled_tools: list[str] | None = None,
        **kwargs,
    ):
        self.system_prompt_path = (
            Path(system_prompt_path) if system_prompt_path else None
        )
        self.disabled_tools = disabled_tools

        super().__init__(
            run_command=_build_run_command(
                agent_workdir,
                disabled_tools=disabled_tools,
                has_system_prompt=system_prompt_path is not None,
            ),
            dataset_path=dataset_path,
            tasks=tasks,
            agent_workdir=agent_workdir,
            docker_image=docker_image,
            **kwargs,
        )

    async def post_sandbox_setup(self, state) -> None:
        """Upload Harbor task assets and optional system prompt after sandbox creation."""
        await super().post_sandbox_setup(state)

        if self.system_prompt_path:
            if not self.system_prompt_path.exists():
                raise FileNotFoundError(
                    f"System prompt file not found: {self.system_prompt_path}"
                )

            sandbox_id = state["sandbox_id"]
            await self.sandbox_client.execute_command(
                sandbox_id, "mkdir -p /opencode", working_dir=None
            )
            await self.sandbox_client.upload_file(
                sandbox_id, "/opencode/prompt.txt", str(self.system_prompt_path)
            )
            logger.info(f"Uploaded system prompt from {self.system_prompt_path}")


def load_environment(
    dataset_path: str | Path = Path(__file__).parent / "tasks",
    dataset: str | None = None,
    tasks: list[str] | None = None,
    agent_workdir: str = "/app",
    docker_image: str = "python:3.11-slim",
    system_prompt_path: str | Path | None = Path(__file__).parent / "prompt.txt",
    disabled_tools: list[str] | None = ["webfetch", "question"],
    timeout_seconds: float = 900.0,
    cpu_cores: int = 2,
    memory_gb: int = 4,
    disk_size_gb: int = 10,
    timeout_minutes: int = 120,
    max_turns: int = 4,
) -> OpenCodeHarborEnv:
    if dataset and tasks:
        raise ValueError("Cannot specify both 'dataset' and 'tasks'")
    if dataset:
        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: {', '.join(DATASETS.keys())}"
            )
        tasks = DATASETS[dataset]

    return OpenCodeHarborEnv(
        dataset_path=dataset_path,
        tasks=tasks,
        agent_workdir=agent_workdir,
        docker_image=docker_image,
        system_prompt_path=system_prompt_path,
        disabled_tools=disabled_tools,
        timeout_seconds=timeout_seconds,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        disk_size_gb=disk_size_gb,
        timeout_minutes=timeout_minutes,
        max_turns=max_turns,
    )
