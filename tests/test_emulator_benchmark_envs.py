import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS_DIR = REPO_ROOT / "environments"
if str(ENVIRONMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENTS_DIR))

from emulator_common.graders import (  # noqa: E402
    audio_crc,
    frame_sequence_sha256,
    framebuffer_sha256,
    stable_within,
    trace_crc,
)
from emulator_common.loader import load_manifest  # noqa: E402
from emulator_common.runners import VERIFIER_SCRIPT  # noqa: E402
from emulator_common.sandbox import build_sandbox_config  # noqa: E402
from emulator_common.suite_adapters import (  # noqa: E402
    ROM_EXTENSIONS_BY_PLATFORM,
    discover_artifacts,
    fetch_source,
    verify_workspace,
)
from emulator_common.test_sources import load_public_test_manifest  # noqa: E402
from verifiers.utils.import_utils import load_toml  # noqa: E402


ENV_SPECS = [
    (
        "emulator_chip8",
        "emulator-chip8",
        1,
        ["IBM Logo ROM", "Corax opcode test", "Timendus CHIP-8 suite"],
    ),
    (
        "emulator_i8080_space_invaders",
        "emulator-i8080-space-invaders",
        2,
        ["8080 CPU exerciser", "Space Invaders attract mode", "Framebuffer hashes"],
    ),
    (
        "emulator_gameboy_dmg",
        "emulator-gameboy-dmg",
        3,
        [
            "blargg test ROMs",
            "mooneye-gb",
            "dmg-acid2",
            "Mealybug Tearoom tests",
            "SameSuite",
        ],
    ),
    (
        "emulator_nes",
        "emulator-nes",
        4,
        [
            "nestest",
            "blargg NES tests",
            "PPU timing tests",
            "Sprite hit tests",
            "VBlank timing tests",
        ],
    ),
    ("emulator_sms", "emulator-sms", 5, ["ZEXDOC", "ZEXALL", "SMS Test Suite"]),
    (
        "emulator_gameboy_cgb",
        "emulator-gameboy-cgb",
        6,
        ["cgb-acid2", "mooneye CGB tests", "SameSuite CGB"],
    ),
    ("emulator_gba", "emulator-gba", 7, ["mGBA suite", "gba-suite", "TONC demos"]),
    (
        "emulator_genesis",
        "emulator-genesis",
        8,
        ["68k verifier", "VDP tests", "Z80 tests", "240p Test Suite"],
    ),
    (
        "emulator_snes",
        "emulator-snes",
        9,
        ["blargg SNES tests", "SNESdev tests", "PPU timing suites"],
    ),
    (
        "emulator_ps1",
        "emulator-ps1",
        10,
        ["ps1-tests", "CPU verification ROMs", "DMA tests", "GPU command tests"],
    ),
]

EXPECTED_ENV_IDS = [env_id for _, env_id, _, _ in ENV_SPECS]


def load_toml_path(path: Path):
    with path.open("rb") as handle:
        return load_toml(handle)


def load_env_module(module_name: str):
    module_path = ENVIRONMENTS_DIR / module_name / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("module_name,env_id,level,public_tests", ENV_SPECS)
def test_manifest_matches_curriculum(module_name, env_id, level, public_tests):
    manifest = load_manifest(ENVIRONMENTS_DIR / module_name / "tasks" / "manifest.json")
    assert manifest.environment_id == env_id
    assert manifest.level == level
    assert list(manifest.public_verification) == public_tests
    assert manifest.requirements
    assert manifest.hidden_verification
    assert manifest.success_criteria
    assert manifest.public_cases()
    assert manifest.hidden_cases()


@pytest.mark.parametrize("module_name,env_id,level,public_tests", ENV_SPECS)
def test_public_test_sources_cover_curriculum(module_name, env_id, level, public_tests):
    _ = level
    source_manifest = load_public_test_manifest(
        ENVIRONMENTS_DIR / module_name / "tasks" / "public_tests.json"
    )
    assert source_manifest.environment_id == env_id
    assert set(public_tests).issubset(source_manifest.covered_names())
    assert all(
        source.url.startswith(("https://", "http://"))
        for source in source_manifest.sources
    )
    assert all(
        source.kind in {"git", "http", "private-artifact"}
        for source in source_manifest.sources
    )


@pytest.mark.parametrize("module_name,env_id,level,public_tests", ENV_SPECS)
def test_environment_entrypoint_loads_taskset(module_name, env_id, level, public_tests):
    _ = (level, public_tests)
    module = load_env_module(module_name)
    env = module.load_environment(
        max_tasks=1,
        sandbox_timeout_minutes=60,
        environment_timeout=60,
        agent_step_limit=5,
        max_turns=1,
    )
    rows = env.taskset.load_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["task_id"] == f"{env_id}__implementation"
    assert row["info"]["environment_id"] == env_id
    assert row["sandbox"]["timeout_minutes"] == 60
    assert row["sandbox"]["command_timeout"] == 3600
    assert "PRIME_TEAM_ID" not in repr(row["sandbox"])


@pytest.mark.parametrize("module_name,env_id,level,public_tests", ENV_SPECS[:1])
def test_environment_eval_smoke_mode_loads_without_sandbox(
    module_name, env_id, level, public_tests
):
    _ = (level, public_tests)
    module = load_env_module(module_name)
    env = module.load_environment(max_tasks=1, eval_smoke=True)
    rows = env.taskset.rows()
    assert len(rows) == 1
    assert rows[0]["task_id"] == f"{env_id}__implementation"
    assert "sandbox" not in rows[0]
    assert "program" not in rows[0]


@pytest.mark.parametrize("module_name,env_id,level,public_tests", ENV_SPECS[:1])
def test_environment_prime_smoke_mode_uses_sandbox_task(
    module_name, env_id, level, public_tests
):
    _ = (level, public_tests)
    module = load_env_module(module_name)
    env = module.load_environment(
        max_tasks=1,
        prime_smoke=True,
        sandbox_timeout_minutes=30,
        environment_timeout=60,
        agent_step_limit=1,
        max_turns=1,
    )
    rows = env.taskset.rows()
    assert len(rows) == 1
    assert rows[0]["task_id"] == f"{env_id}__prime_smoke"
    assert rows[0]["sandbox"]["timeout_minutes"] == 30
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in rows[0]["question"]


def test_implementation_harness_scores_after_agent_command_failure():
    module = load_env_module("emulator_chip8")
    env = module.load_environment(
        max_tasks=1,
        sandbox_timeout_minutes=30,
        environment_timeout=60,
        agent_step_limit=1,
        max_turns=1,
    )

    program = env.harness.program
    assert isinstance(program, dict)
    command = program["command"]
    assert isinstance(command, list)
    assert "emulator_agent_exit_status" in command[2]
    assert "running verifier anyway" in command[2]
    assert command[2].rstrip().endswith("exit 0")
    assert program["artifacts"]["emulator_agent_exit_status"]["path"] == (
        "/tmp/emulator_agent_exit_status"
    )


def test_inline_system_prompt_mode_uses_user_message_for_prime_rl():
    module = load_env_module("emulator_chip8")
    env = module.load_environment(
        max_tasks=1,
        sandbox_timeout_minutes=30,
        environment_timeout=60,
        agent_step_limit=1,
        max_turns=1,
        inline_system_prompt=True,
    )

    row = env.taskset.load_rows()[0]
    assert row["prompt"] == [{"role": "user", "content": row["instruction"]}]
    assert row["instruction"].startswith(
        "You are implementing a hardware emulator for a deterministic coding benchmark."
    )
    assert getattr(env.harness, "system_prompt", None) == []


def test_prime_smoke_harness_preserves_agent_command_failure():
    module = load_env_module("emulator_chip8")
    env = module.load_environment(
        max_tasks=1,
        prime_smoke=True,
        sandbox_timeout_minutes=30,
        environment_timeout=60,
        agent_step_limit=1,
        max_turns=1,
    )

    program = env.harness.program
    assert isinstance(program, dict)
    command = program["command"]
    assert isinstance(command, list)
    assert "emulator_agent_exit_status" not in command[2]
    assert "emulator_agent_exit_status" not in program["artifacts"]


def test_sandbox_uses_prime_toolchain_env(monkeypatch):
    monkeypatch.setenv("PRIME_TOOLCHAIN_IMAGE", "toolchain/from-env:latest")
    config = build_sandbox_config(timeout_minutes=360)
    assert config["image"] == "toolchain/from-env:latest"
    assert config["timeout_minutes"] == 360
    assert config["command_timeout"] == 21600
    assert config["network_access"] is True


def test_grader_helpers_are_deterministic():
    assert framebuffer_sha256([1, 2, 3]) == framebuffer_sha256(bytes([1, 2, 3]))
    assert frame_sequence_sha256([[1], [2, 3]]) == frame_sequence_sha256(
        [b"\x01", b"\x02\x03"]
    )
    assert trace_crc([{"pc": 1, "a": 2}]) == trace_crc([{"a": 2, "pc": 1}])
    assert audio_crc([0, 255, 1]) == audio_crc(bytes([0, 255, 1]))
    assert stable_within([59.9, 60.0, 60.1], 0.2)
    assert not stable_within([59.0, 61.0], 0.2)


def test_hidden_verifier_script_is_valid_python():
    compile(VERIFIER_SCRIPT, "emulator_verify.py", "exec")


def test_test_source_verifier_script_is_valid_python():
    script = ENVIRONMENTS_DIR / "emulator_common" / "scripts" / "verify_test_sources.py"
    compile(script.read_text(encoding="utf-8"), str(script), "exec")


def test_gpt54_smoke_eval_config_covers_all_emulator_envs():
    config = load_toml_path(
        REPO_ROOT / "configs" / "eval" / "emulator-gpt-5-4-mini-prime-smoke.toml"
    )
    assert config["model"] == "openai/gpt-5.4-mini"
    assert config["num_examples"] == 1
    assert config["rollouts_per_example"] == 1

    evals = config["eval"]
    assert [row["id"] for row in evals] == EXPECTED_ENV_IDS
    assert all(row["args"]["prime_smoke"] is True for row in evals)
    assert all(row["args"]["agent_step_limit"] == 1 for row in evals)
    assert all(row["args"]["max_turns"] == 1 for row in evals)


def test_gpt54_suite_eval_config_covers_all_emulator_envs():
    config = load_toml_path(
        REPO_ROOT / "configs" / "eval" / "emulator-gpt-5-4-mini-suite.toml"
    )
    assert config["model"] == "openai/gpt-5.4-mini"
    assert config["max_concurrent"] == 1
    assert [row["id"] for row in config["eval"]] == EXPECTED_ENV_IDS
    assert all("prime_smoke" not in row["args"] for row in config["eval"])


def test_chip8_suite_canary_config_matches_recorded_result_scope():
    config = load_toml_path(
        REPO_ROOT / "configs" / "eval" / "emulator-gpt-5-4-mini-chip8-suite.toml"
    )
    assert config["model"] == "openai/gpt-5.4-mini"
    assert [row["id"] for row in config["eval"]] == ["emulator-chip8"]
    assert config["eval"][0]["args"]["agent_step_limit"] == 1


def test_minimal_emulator_training_config_uses_trainable_model():
    config = load_toml_path(
        REPO_ROOT / "configs" / "rl" / "emulator-chip8-minimal.toml"
    )
    assert config["model"] == "openai/gpt-oss-20b"
    assert config["max_steps"] == 10
    assert config["batch_size"] == 8
    assert config["rollouts_per_example"] == 2
    assert config["env"][0]["id"] == "emulator-chip8"
    assert config["env"][0]["args"]["max_tasks"] == 1


def test_self_managed_prime_rl_smoke_config_targets_two_gpu_node():
    config = load_toml_path(
        REPO_ROOT / "configs" / "rl" / "emulator-chip8-prime-rl-smoke.toml"
    )

    assert config["model"]["name"] == "Qwen/Qwen3-0.6B"
    assert config["deployment"]["gpus_per_node"] == 2
    assert config["deployment"]["num_train_gpus"] == 1
    assert config["deployment"]["num_infer_gpus"] == 1
    assert config["orchestrator"]["use_token_client"] is False
    assert config["orchestrator"]["use_renderer"] is True
    assert config["orchestrator"]["filters"][-1] == {
        "type": "zero_advantage",
        "enforce": False,
    }
    assert config["orchestrator"]["batch_size"] == 1
    assert config["orchestrator"]["train"]["env"][0]["id"] == "emulator-chip8"
    env_args = config["orchestrator"]["train"]["env"][0]["args"]
    assert env_args["max_tasks"] == 1
    assert env_args["inline_system_prompt"] is True
    assert config["trainer"]["max_steps"] == 1


def test_git_source_sparse_paths_and_compressed_artifacts(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "wanted").mkdir()
    (repo / "other").mkdir()
    (repo / "wanted" / "ADD.json.gz").write_bytes(b"processor fixture")
    (repo / "other" / "ignored.json.gz").write_bytes(b"ignored fixture")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "commit", "-m", "fixtures"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    )

    destination = tmp_path / "cache"
    result = fetch_source(
        {
            "name": "processor tests",
            "kind": "git",
            "url": str(repo),
            "sparse_paths": ["wanted"],
            "artifact_globs": ["wanted/*.json.gz"],
            "artifact_extensions": [".json.gz"],
        },
        destination,
        timeout=30,
    )

    assert result["ok"] is True
    artifacts = discover_artifacts("genesis", [Path(str(result["path"]))])
    assert [artifact.name for artifact in artifacts] == ["ADD.json.gz"]


@pytest.mark.parametrize("module_name,env_id,level,public_tests", ENV_SPECS)
def test_suite_adapter_runs_minimal_fixture(
    tmp_path: Path, module_name, env_id, level, public_tests
):
    _ = (env_id, level, public_tests)
    manifest = load_manifest(ENVIRONMENTS_DIR / module_name / "tasks" / "manifest.json")
    platform = manifest.slug
    extension = ROM_EXTENSIONS_BY_PLATFORM[platform][0]

    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    (suite_dir / f"fixture{extension}").write_bytes(b"fixture-rom")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = workspace / "emulator-runner"
    runner.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import argparse
            import hashlib
            import json
            import sys

            if "--contract-version" in sys.argv:
                print("emulator-runner-v1")
                raise SystemExit(0)
            if "--self-test" in sys.argv:
                print(json.dumps({"ok": True}))
                raise SystemExit(0)

            parser = argparse.ArgumentParser()
            parser.add_argument("--platform", required=True)
            parser.add_argument("--case", required=True)
            parser.add_argument("--rom", required=True)
            parser.add_argument("--cycles", required=True)
            parser.add_argument("--frames", required=True)
            parser.add_argument("--output", required=True)
            args = parser.parse_args()
            rom = open(args.rom, "rb").read()
            digest = hashlib.sha256(rom + args.case.encode()).hexdigest()
            payload = {
                "platform": args.platform,
                "case_id": args.case,
                "passed": True,
                "cycles": int(args.cycles),
                "frames": int(args.frames),
                "framebuffer_sha256": digest,
                "frame_sequence_sha256": digest,
                "trace_crc32": digest[:8],
                "audio_crc32": digest[:8],
                "serial": "Passed",
                "perf": {"fps": 60.0},
            }
            open(args.output, "w", encoding="utf-8").write(json.dumps(payload))
            print(json.dumps({"passed": True}))
            """
        ),
        encoding="utf-8",
    )
    runner.chmod(0o755)

    source_manifest = {
        "environment_id": manifest.environment_id,
        "sources": [
            {
                "name": "fixture suite",
                "kind": "local",
                "url": str(suite_dir),
                "covers": list(manifest.public_verification),
            }
        ],
        "private_artifacts": [],
    }
    result = verify_workspace(
        workspace,
        {
            "platform": platform,
            "public_cases": manifest.public_cases(),
            "runtime": dict(manifest.runtime),
        },
        source_manifest,
        skip_build=True,
        timeout=30,
    )
    assert result["components"]["runner_contract_score"] == 1.0
    assert result["components"]["public_source_score"] == 1.0
    assert result["components"]["public_rom_pass_rate"] == 1.0
    assert result["components"]["deterministic_replay_score"] == 1.0
    assert result["score"] >= 0.9
