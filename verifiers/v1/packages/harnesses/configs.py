from collections.abc import Mapping

from pydantic import field_validator

from ...config import HarnessConfig, PromptInput, SandboxConfig

OPENCODE_DEFAULT_RELEASE_REPO = "PrimeIntellect-ai/opencode"
OPENCODE_DEFAULT_RELEASE_VERSION = "1.1.63-rl2"
OPENCODE_DEFAULT_RELEASE_SHA256 = (
    "47f4102796da50769e27d2c9ea6a9cf7941f76898390cb497278cab39c4b6ed4"
)
OPENCODE_DEFAULT_AGENT_WORKDIR = "/app"
OPENCODE_DEFAULT_INSTRUCTION_PATH = "/opencode/instruction.txt"
OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH = "/opencode/system.txt"
OPENCODE_DEFAULT_LOG_PATH = "/logs/agent/opencode.txt"
OPENCODE_DEFAULT_SYSTEM_PROMPT = """\
You are OpenCode, an interactive CLI tool that helps users with tasks.

Your output is displayed in a command line interface. Be concise and direct.
Use tools to complete tasks. Do not use shell commands or code comments as a
way to communicate with the user.
"""
OPENCODE_DEFAULT_DISABLED_TOOLS = [
    "apply_patch",
    "write",
    "multiedit",
    "glob",
    "todowrite",
    "todoread",
    "websearch",
    "task",
    "batch",
    "list",
    "read",
    "question",
    "webfetch",
    "grep",
    "plan_exit",
    "plan_enter",
    "lsp",
    "codesearch",
    "skill",
]

MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR = "${AGENT_WORKDIR:-/app}"
MINI_SWE_AGENT_DEFAULT_INSTRUCTION_PATH = "/mini-swe-agent/prompt.txt"
MINI_SWE_AGENT_DEFAULT_SYSTEM_PROMPT_PATH = "/mini-swe-agent/system.txt"
MINI_SWE_AGENT_DEFAULT_LOG_PATH = "/logs/agent/mini-swe-agent.log"
MINI_SWE_AGENT_DEFAULT_TRAJECTORY_PATH = "/logs/agent/mini-swe-agent.traj.json"
MINI_SWE_AGENT_DEFAULT_PACKAGE_VERSION = "2.2.8"
MINI_SWE_AGENT_DEFAULT_PACKAGE_SHA256 = (
    "694df4de1337e665e3cd82e99f93374f573bf52b8e7c362ac5d8045ad9f7c37c"
)
MINI_SWE_AGENT_DEFAULT_CONFIG_SPEC = "mini_textbased"
MINI_SWE_AGENT_DEFAULT_MODEL_CLASS = "litellm_textbased"
MINI_SWE_AGENT_DEFAULT_ENVIRONMENT_TIMEOUT = 120

PI_DEFAULT_PACKAGE = "@mariozechner/pi-coding-agent"
PI_DEFAULT_WORKDIR = "/app"
PI_DEFAULT_INSTRUCTION_PATH = "/pi/instruction.txt"
PI_DEFAULT_SYSTEM_PROMPT_PATH = "/pi/system.txt"
PI_DEFAULT_LOG_PATH = "/logs/agent/pi.txt"
PI_DEFAULT_SYSTEM_PROMPT = "Complete the user's task using the available tools."

RLM_DEFAULT_REPO_URL = "github.com/PrimeIntellect-ai/rlm-harness.git"
RLM_DEFAULT_REPO_REF = "main"
RLM_DEFAULT_MAX_TURNS = 100
RLM_DEFAULT_EXEC_TIMEOUT = 300
RLM_DEFAULT_MAX_DEPTH = 0
RLM_DEFAULT_INSTRUCTION_PATH = "/rlm/instruction.txt"
RLM_DEFAULT_APPEND_TO_SYSTEM_PROMPT_PATH = "/rlm/append_to_system_prompt.txt"
RLM_DEFAULT_WORKDIR = "/workspace"
RLM_DEFAULT_TOOLS = ["ipython"]

TERMINUS_2_DEFAULT_AGENT_WORKDIR = "/app"
TERMINUS_2_DEFAULT_INSTRUCTION_PATH = "/terminus_2/instruction.md"
TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH = "/terminus_2/system_prompt.txt"
TERMINUS_2_DEFAULT_LOG_PATH = "/logs/agent/terminus_2.log"
TERMINUS_2_DEFAULT_HARBOR_PACKAGE = "harbor==0.6.6"
TERMINUS_2_DEFAULT_PYTHON_VERSION = "3.12"
TERMINUS_2_DEFAULT_MODEL_NAME = "openai/gpt-4.1-mini"
TERMINUS_2_DEFAULT_API_BASE_URL = "https://api.pinference.ai/api/v1"


class OpenCodeConfig(HarnessConfig):
    agent_workdir: str = OPENCODE_DEFAULT_AGENT_WORKDIR
    instruction_path: str = OPENCODE_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = OPENCODE_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = OPENCODE_DEFAULT_LOG_PATH
    system_prompt: PromptInput | None = OPENCODE_DEFAULT_SYSTEM_PROMPT
    disabled_tools: list[str] = OPENCODE_DEFAULT_DISABLED_TOOLS
    allow_git: bool = False
    disable_compaction: bool = True
    release_repo: str = OPENCODE_DEFAULT_RELEASE_REPO
    release_version: str = OPENCODE_DEFAULT_RELEASE_VERSION
    release_sha256: str = OPENCODE_DEFAULT_RELEASE_SHA256
    install_ripgrep: bool = True
    provider_timeout_ms: int = 3_600_000
    max_turns: int = 4


class MiniSWEAgentConfig(HarnessConfig):
    agent_workdir: str = MINI_SWE_AGENT_DEFAULT_AGENT_WORKDIR
    instruction_path: str = MINI_SWE_AGENT_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = MINI_SWE_AGENT_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = MINI_SWE_AGENT_DEFAULT_LOG_PATH
    trajectory_path: str = MINI_SWE_AGENT_DEFAULT_TRAJECTORY_PATH
    package_version: str = MINI_SWE_AGENT_DEFAULT_PACKAGE_VERSION
    package_sha256: str = MINI_SWE_AGENT_DEFAULT_PACKAGE_SHA256
    config_spec: str = MINI_SWE_AGENT_DEFAULT_CONFIG_SPEC
    model_class: str = MINI_SWE_AGENT_DEFAULT_MODEL_CLASS
    environment_timeout: int = MINI_SWE_AGENT_DEFAULT_ENVIRONMENT_TIMEOUT
    extra_config_specs: list[str] | None = None
    install_python: bool = True
    system_prompt: PromptInput | None = None
    sandbox: SandboxConfig | None = SandboxConfig()
    max_turns: int = 4


class PiConfig(HarnessConfig):
    agent_workdir: str = PI_DEFAULT_WORKDIR
    instruction_path: str = PI_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = PI_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = PI_DEFAULT_LOG_PATH
    system_prompt: PromptInput | None = PI_DEFAULT_SYSTEM_PROMPT
    package: str = PI_DEFAULT_PACKAGE
    install_mcp_adapter: bool = True
    sandbox: SandboxConfig | None = SandboxConfig()
    max_turns: int = 4


class RLMConfig(HarnessConfig):
    workdir: str = RLM_DEFAULT_WORKDIR
    instruction_path: str = RLM_DEFAULT_INSTRUCTION_PATH
    rlm_repo_url: str = RLM_DEFAULT_REPO_URL
    rlm_repo_ref: str = RLM_DEFAULT_REPO_REF
    rlm_max_turns: int = RLM_DEFAULT_MAX_TURNS
    rlm_exec_timeout: int = RLM_DEFAULT_EXEC_TIMEOUT
    rlm_max_depth: int = RLM_DEFAULT_MAX_DEPTH
    summarize_at_tokens: int | tuple[int, int] | list[int] | None = None
    include_sub_rlm_trajectories: bool = False
    append_to_system_prompt: str = ""
    local_checkout: str | None = None
    gh_token: str | None = None
    rlm_tools: list[str] = RLM_DEFAULT_TOOLS
    env_vars: dict[str, str] = {}
    skills: str | None = None

    @field_validator("env_vars", mode="before")
    @classmethod
    def validate_env_vars(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): str(item) for key, item in value.items()}
        return value


class Terminus2Config(HarnessConfig):
    agent_workdir: str = TERMINUS_2_DEFAULT_AGENT_WORKDIR
    instruction_path: str = TERMINUS_2_DEFAULT_INSTRUCTION_PATH
    system_prompt_path: str = TERMINUS_2_DEFAULT_SYSTEM_PROMPT_PATH
    log_path: str = TERMINUS_2_DEFAULT_LOG_PATH
    harbor_package: str = TERMINUS_2_DEFAULT_HARBOR_PACKAGE
    python_version: str = TERMINUS_2_DEFAULT_PYTHON_VERSION
    model_name: str = TERMINUS_2_DEFAULT_MODEL_NAME
    api_base_url: str = TERMINUS_2_DEFAULT_API_BASE_URL
    system_prompt: PromptInput | None = None
    sandbox: SandboxConfig | None = SandboxConfig()
    max_turns: int = 4
