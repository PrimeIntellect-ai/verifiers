"""Tests for the OpenCodeRLMEnv class."""

import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

from verifiers.envs.experimental.opencode_rlm_env import (
    OpenCodeRLMEnv,
    OpenCodeRLMMonitorRubric,
)


# =============================================================================
# Helpers
# =============================================================================


def make_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "hello"}]],
            "answer": ["world"],
        }
    )


def build_env(**kwargs) -> OpenCodeRLMEnv:
    kwargs.setdefault("dataset", make_dataset())
    with patch("verifiers.envs.environment.signal.signal"):
        return OpenCodeRLMEnv(**kwargs)


# =============================================================================
# Constructor defaults
# =============================================================================


class TestConstructorDefaults:
    def test_default_plugin_repo(self):
        env = build_env()
        assert env.plugin_repo == "snimu/oc"

    def test_default_plugin_branch(self):
        env = build_env()
        assert env.plugin_branch == "main"

    def test_default_plugin_install_path(self):
        env = build_env()
        assert env.plugin_install_path == "/tmp/opencode-rlm"

    def test_default_sub_model_identifier(self):
        env = build_env()
        assert env.sub_model_identifier == "sub"

    def test_default_sub_model(self):
        env = build_env()
        assert env.sub_model is None

    def test_default_sub_llm_max_turns(self):
        env = build_env()
        assert env.sub_llm_max_turns == 10

    def test_default_sub_timeout_ms(self):
        env = build_env()
        assert env.sub_timeout_ms == 120_000

    def test_custom_params(self):
        env = build_env(
            plugin_repo="org/repo",
            plugin_branch="dev",
            plugin_install_path="/opt/plugin",
            sub_model_identifier="child",
            sub_model="gpt-4o-mini",
            sub_llm_max_turns=5,
            sub_timeout_ms=60_000,
            include_sub_llm_in_trajectory=True,
        )
        assert env.plugin_repo == "org/repo"
        assert env.plugin_branch == "dev"
        assert env.plugin_install_path == "/opt/plugin"
        assert env.sub_model_identifier == "child"
        assert env.sub_model == "gpt-4o-mini"
        assert env.sub_llm_max_turns == 5
        assert env.sub_timeout_ms == 60_000
        assert env.include_sub_llm_in_trajectory is True


# =============================================================================
# OpenCode config
# =============================================================================


class TestOpenCodeConfig:
    def test_config_includes_plugin_reference(self):
        env = build_env()
        config_str = env.build_opencode_config()
        config = json.loads(config_str)
        assert "plugin" in config
        assert config["plugin"] == ["file:///tmp/opencode-rlm"]

    def test_config_custom_install_path(self):
        env = build_env(plugin_install_path="/custom/path")
        config = json.loads(env.build_opencode_config())
        assert config["plugin"] == ["file:///custom/path"]

    def test_config_has_schema_and_provider(self):
        env = build_env()
        config = json.loads(env.build_opencode_config())
        assert "$schema" in config or "${SCHEMA_DOLLAR}schema" in json.dumps(config)
        assert "provider" in config
        assert "model" in config

    def test_config_renders_valid_json_after_shell_expansion(self):
        env = build_env()
        run_command = env.run_command

        prefix = "cat > ~/.config/opencode/opencode.json << EOFCONFIG\n"
        suffix = "\nEOFCONFIG"
        assert prefix in run_command, "Config block not found in run command"
        config_block = run_command.split(prefix, 1)[1].split(suffix, 1)[0]

        script = (
            f"OPENAI_MODEL=openai/gpt-5-mini "
            f"OPENAI_BASE_URL=https://example.invalid "
            f"SCHEMA_DOLLAR='$' "
            f"bash -lc 'cat <<EOFCONFIG\n{config_block}\nEOFCONFIG'"
        )
        result = subprocess.run(
            script,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            check=True,
        )
        config = json.loads(result.stdout)
        assert config["$schema"] == "https://opencode.ai/config.json"
        assert config["plugin"] == ["file:///tmp/opencode-rlm"]
        assert (
            config["provider"]["openai"]["options"]["baseURL"]
            == "https://example.invalid"
        )


# =============================================================================
# Run command
# =============================================================================


class TestRunCommand:
    def test_run_command_installs_jq(self):
        env = build_env()
        assert "apt-get install -y curl git unzip jq" in env.run_command

    def test_run_command_installs_bun(self):
        env = build_env()
        assert "bun.sh/install" in env.run_command

    def test_run_command_clones_plugin(self):
        env = build_env()
        assert (
            "git clone --branch main https://github.com/snimu/oc.git /tmp/opencode-rlm"
            in env.run_command
        )

    def test_run_command_custom_branch(self):
        env = build_env(plugin_branch="feature/x")
        assert "git clone --branch feature/x" in env.run_command

    def test_run_command_bun_install(self):
        env = build_env()
        assert "cd /tmp/opencode-rlm && bun install" in env.run_command

    def test_run_command_runs_opencode(self):
        env = build_env()
        assert "opencode run" in env.run_command


# =============================================================================
# Environment variables
# =============================================================================


class TestBuildEnvVars:
    @pytest.mark.asyncio
    async def test_sets_sub_model_id(self):
        env = build_env()
        state = {
            "interception_base_url": "http://localhost:8080",
            "model": "openai/gpt-5-mini",
        }
        env_vars = await env.build_env_vars(state)
        assert env_vars["RLM_SUB_MODEL_ID"] == "sub"

    @pytest.mark.asyncio
    async def test_sets_proxy_mode(self):
        env = build_env()
        state = {
            "interception_base_url": "http://localhost:8080",
            "model": "openai/gpt-5-mini",
        }
        env_vars = await env.build_env_vars(state)
        assert env_vars["RLM_LLM_SUBCALL_VIA_PROXY"] == "true"

    @pytest.mark.asyncio
    async def test_sets_tool_loop_mode(self):
        env = build_env()
        state = {
            "interception_base_url": "http://localhost:8080",
            "model": "openai/gpt-5-mini",
        }
        env_vars = await env.build_env_vars(state)
        assert env_vars["RLM_SUBAGENT_VIA_TOOL_LOOP"] == "true"

    @pytest.mark.asyncio
    async def test_sets_max_turns(self):
        env = build_env(sub_llm_max_turns=5)
        state = {
            "interception_base_url": "http://localhost:8080",
            "model": "openai/gpt-5-mini",
        }
        env_vars = await env.build_env_vars(state)
        assert env_vars["RLM_SUB_MAX_TURNS"] == "5"

    @pytest.mark.asyncio
    async def test_sets_timeout(self):
        env = build_env(sub_timeout_ms=60_000)
        state = {
            "interception_base_url": "http://localhost:8080",
            "model": "openai/gpt-5-mini",
        }
        env_vars = await env.build_env_vars(state)
        assert env_vars["RLM_SUB_TIMEOUT"] == "60000"

    @pytest.mark.asyncio
    async def test_preserves_base_env_vars(self):
        env = build_env()
        state = {
            "interception_base_url": "http://localhost:8080",
            "model": "openai/gpt-5-mini",
        }
        env_vars = await env.build_env_vars(state)
        assert "OPENAI_BASE_URL" in env_vars
        assert env_vars["OPENAI_BASE_URL"] == "http://localhost:8080"

    @pytest.mark.asyncio
    async def test_custom_sub_model_identifier(self):
        env = build_env(sub_model_identifier="child-model")
        state = {
            "interception_base_url": "http://localhost:8080",
            "model": "openai/gpt-5-mini",
        }
        env_vars = await env.build_env_vars(state)
        assert env_vars["RLM_SUB_MODEL_ID"] == "child-model"


# =============================================================================
# Sub-LLM detection
# =============================================================================


class TestIsSubLLMRequest:
    def test_detects_sub_model(self):
        env = build_env()
        assert env._is_sub_llm_request({"model": "sub"}) is True

    def test_detects_sub_in_compound_model(self):
        env = build_env()
        assert env._is_sub_llm_request({"model": "sub/gpt-4o"}) is True

    def test_rejects_main_model(self):
        env = build_env()
        assert env._is_sub_llm_request({"model": "openai/gpt-5-mini"}) is False

    def test_rejects_empty_model(self):
        env = build_env()
        assert env._is_sub_llm_request({"model": ""}) is False

    def test_rejects_missing_model(self):
        env = build_env()
        assert env._is_sub_llm_request({}) is False

    def test_custom_identifier(self):
        env = build_env(sub_model_identifier="child")
        assert env._is_sub_llm_request({"model": "child"}) is True
        assert env._is_sub_llm_request({"model": "sub"}) is False


# =============================================================================
# State setup
# =============================================================================


class TestSetupState:
    @pytest.mark.asyncio
    async def test_initializes_metrics(self):
        env = build_env()
        state: dict = {}
        # Mock super().setup_state to just return state
        with patch.object(
            OpenCodeRLMEnv.__bases__[0],
            "setup_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await env.setup_state(state)
        assert result["main_turns"] == 0
        assert result["main_prompt_tokens"] == 0
        assert result["main_completion_tokens"] == 0
        assert result["sub_llm_turns"] == 0
        assert result["sub_llm_prompt_tokens"] == 0
        assert result["sub_llm_completion_tokens"] == 0

    @pytest.mark.asyncio
    async def test_preserves_existing_metrics(self):
        env = build_env()
        state: dict = {"main_turns": 5, "sub_llm_turns": 3}
        with patch.object(
            OpenCodeRLMEnv.__bases__[0],
            "setup_state",
            new_callable=AsyncMock,
            return_value=state,
        ):
            result = await env.setup_state(state)
        assert result["main_turns"] == 5
        assert result["sub_llm_turns"] == 3


# =============================================================================
# Metrics helpers
# =============================================================================


class TestMetrics:
    def test_extract_token_counts(self):
        response = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        assert OpenCodeRLMEnv._extract_token_counts(response) == (100, 50)

    def test_extract_token_counts_no_usage(self):
        response = MagicMock(spec=[])  # no usage attr
        assert OpenCodeRLMEnv._extract_token_counts(response) == (0, 0)

    def test_update_main_metrics(self):
        env = build_env()
        state = {
            "main_turns": 1,
            "main_prompt_tokens": 100,
            "main_completion_tokens": 50,
        }
        response = MagicMock()
        response.usage.prompt_tokens = 200
        response.usage.completion_tokens = 80
        env._update_main_metrics(state, response)
        assert state["main_turns"] == 2
        assert state["main_prompt_tokens"] == 300
        assert state["main_completion_tokens"] == 130

    def test_update_sub_metrics(self):
        env = build_env()
        state = {
            "sub_llm_turns": 0,
            "sub_llm_prompt_tokens": 0,
            "sub_llm_completion_tokens": 0,
        }
        response = MagicMock()
        response.usage.prompt_tokens = 50
        response.usage.completion_tokens = 20
        env._update_sub_metrics(state, response)
        assert state["sub_llm_turns"] == 1
        assert state["sub_llm_prompt_tokens"] == 50
        assert state["sub_llm_completion_tokens"] == 20


# =============================================================================
# Monitor rubric
# =============================================================================


class TestMonitorRubric:
    def test_rubric_has_all_metrics(self):
        rubric = OpenCodeRLMMonitorRubric()
        func_names = {f.__name__ for f in rubric.funcs}
        expected = {
            "main_turns",
            "main_prompt_tokens",
            "main_completion_tokens",
            "sub_llm_turns",
            "sub_llm_prompt_tokens",
            "sub_llm_completion_tokens",
        }
        assert expected.issubset(func_names)

    @pytest.mark.asyncio
    async def test_metric_reads_from_state(self):
        rubric = OpenCodeRLMMonitorRubric()
        state = {"main_turns": 7, "sub_llm_turns": 3}
        assert await rubric.main_turns(state) == 7.0
        assert await rubric.sub_llm_turns(state) == 3.0

    @pytest.mark.asyncio
    async def test_metric_defaults_to_zero(self):
        rubric = OpenCodeRLMMonitorRubric()
        assert await rubric.main_turns({}) == 0.0
