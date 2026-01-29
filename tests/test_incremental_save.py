"""Tests for incremental save and resume functionality.

Covers:
- append_outputs_to_disk: Appending outputs to results.jsonl
- read_existing_outputs: Reading outputs from results.jsonl
- read_completed_example_ids: Reading completed example IDs for resume
- read_completed_rollout_ids: Reading completed rollout IDs for resume
- GenerateOutputsBuilder resume mode: Loading existing outputs
- find_resumable_run: Finding runs to resume from
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from verifiers.types import RolloutOutput
from verifiers.utils.path_utils import find_eval_runs, find_resumable_run
from verifiers.utils.save_utils import (
    GenerateOutputsBuilder,
    append_outputs_to_disk,
    read_completed_example_ids,
    read_completed_rollout_ids,
    read_existing_outputs,
)


class TestAppendOutputsToDisk:
    def test_append_creates_file_if_not_exists(self, tmp_path):
        """Test that append_outputs_to_disk creates the file if it doesn't exist."""
        outputs = [
            RolloutOutput(
                example_id=0,
                task="test",
                prompt=[{"role": "user", "content": "test"}],
                completion=[{"role": "assistant", "content": "response"}],
                reward=1.0,
                timing={},
                is_completed=True,
                is_truncated=False,
                metrics={},
                rollout_idx=0,
            )
        ]

        append_outputs_to_disk(outputs, tmp_path)

        results_file = tmp_path / "results.jsonl"
        assert results_file.exists()
        with open(results_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["example_id"] == 0
        assert data["rollout_idx"] == 0

    def test_append_adds_to_existing_file(self, tmp_path):
        """Test that append_outputs_to_disk appends to existing file."""
        results_file = tmp_path / "results.jsonl"

        # First write
        outputs1 = [
            RolloutOutput(
                example_id=0,
                task="test",
                prompt=[],
                completion=[],
                reward=1.0,
                timing={},
                is_completed=True,
                is_truncated=False,
                metrics={},
                rollout_idx=0,
            )
        ]
        append_outputs_to_disk(outputs1, tmp_path)

        # Second write
        outputs2 = [
            RolloutOutput(
                example_id=1,
                task="test",
                prompt=[],
                completion=[],
                reward=0.5,
                timing={},
                is_completed=True,
                is_truncated=False,
                metrics={},
                rollout_idx=0,
            )
        ]
        append_outputs_to_disk(outputs2, tmp_path)

        # Check both are present
        with open(results_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["example_id"] == 0
        assert json.loads(lines[1])["example_id"] == 1


class TestReadExistingOutputs:
    def test_read_empty_file(self, tmp_path):
        """Test reading from an empty file."""
        results_file = tmp_path / "results.jsonl"
        results_file.touch()

        outputs = read_existing_outputs(results_file)
        assert outputs == []

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading from a nonexistent file."""
        results_file = tmp_path / "results.jsonl"

        outputs = read_existing_outputs(results_file)
        assert outputs == []

    def test_read_existing_outputs(self, tmp_path):
        """Test reading outputs from a file."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write(
                json.dumps({"example_id": 0, "reward": 1.0, "rollout_idx": 0}) + "\n"
            )
            f.write(
                json.dumps({"example_id": 1, "reward": 0.5, "rollout_idx": 0}) + "\n"
            )

        outputs = read_existing_outputs(results_file)
        assert len(outputs) == 2
        assert outputs[0]["example_id"] == 0
        assert outputs[1]["example_id"] == 1

    def test_read_handles_malformed_json(self, tmp_path):
        """Test that malformed JSON lines are skipped with warning."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write(json.dumps({"example_id": 0, "reward": 1.0}) + "\n")
            f.write("invalid json\n")
            f.write(json.dumps({"example_id": 1, "reward": 0.5}) + "\n")

        outputs = read_existing_outputs(results_file)
        assert len(outputs) == 2  # malformed line skipped


class TestReadCompletedExampleIds:
    def test_empty_file(self, tmp_path):
        """Test with empty file."""
        results_file = tmp_path / "results.jsonl"
        results_file.touch()

        completed = read_completed_example_ids(results_file, rollouts_per_example=2)
        assert completed == set()

    def test_nonexistent_file(self, tmp_path):
        """Test with nonexistent file."""
        results_file = tmp_path / "results.jsonl"

        completed = read_completed_example_ids(results_file, rollouts_per_example=2)
        assert completed == set()

    def test_incomplete_examples(self, tmp_path):
        """Test that incomplete examples are not returned."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            # Only 1 rollout for example 0, but we need 2
            f.write(json.dumps({"example_id": 0, "rollout_idx": 0}) + "\n")

        completed = read_completed_example_ids(results_file, rollouts_per_example=2)
        assert completed == set()

    def test_complete_examples(self, tmp_path):
        """Test that complete examples are returned."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            # 2 rollouts for example 0
            f.write(json.dumps({"example_id": 0, "rollout_idx": 0}) + "\n")
            f.write(json.dumps({"example_id": 0, "rollout_idx": 1}) + "\n")
            # 1 rollout for example 1 (incomplete)
            f.write(json.dumps({"example_id": 1, "rollout_idx": 0}) + "\n")

        completed = read_completed_example_ids(results_file, rollouts_per_example=2)
        assert completed == {0}

    def test_mixed_complete_incomplete(self, tmp_path):
        """Test with a mix of complete and incomplete examples."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            # 3 rollouts for example 0 (complete)
            f.write(json.dumps({"example_id": 0}) + "\n")
            f.write(json.dumps({"example_id": 0}) + "\n")
            f.write(json.dumps({"example_id": 0}) + "\n")
            # 3 rollouts for example 1 (complete)
            f.write(json.dumps({"example_id": 1}) + "\n")
            f.write(json.dumps({"example_id": 1}) + "\n")
            f.write(json.dumps({"example_id": 1}) + "\n")
            # 2 rollouts for example 2 (incomplete)
            f.write(json.dumps({"example_id": 2}) + "\n")
            f.write(json.dumps({"example_id": 2}) + "\n")

        completed = read_completed_example_ids(results_file, rollouts_per_example=3)
        assert completed == {0, 1}


class TestReadCompletedRolloutIds:
    def test_empty_file(self, tmp_path):
        """Test with empty file."""
        results_file = tmp_path / "results.jsonl"
        results_file.touch()

        completed = read_completed_rollout_ids(results_file)
        assert completed == set()

    def test_nonexistent_file(self, tmp_path):
        """Test with nonexistent file."""
        results_file = tmp_path / "results.jsonl"

        completed = read_completed_rollout_ids(results_file)
        assert completed == set()

    def test_read_rollout_ids(self, tmp_path):
        """Test reading (example_id, rollout_idx) pairs."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write(json.dumps({"example_id": 0, "rollout_idx": 0}) + "\n")
            f.write(json.dumps({"example_id": 0, "rollout_idx": 1}) + "\n")
            f.write(json.dumps({"example_id": 1, "rollout_idx": 0}) + "\n")

        completed = read_completed_rollout_ids(results_file)
        assert completed == {(0, 0), (0, 1), (1, 0)}

    def test_default_rollout_idx(self, tmp_path):
        """Test that missing rollout_idx defaults to 0."""
        results_file = tmp_path / "results.jsonl"
        with open(results_file, "w") as f:
            f.write(json.dumps({"example_id": 0}) + "\n")
            f.write(json.dumps({"example_id": 1}) + "\n")

        completed = read_completed_rollout_ids(results_file)
        assert completed == {(0, 0), (1, 0)}


class TestGenerateOutputsBuilderResume:
    def test_load_existing_outputs(self, make_output):
        """Test loading existing outputs into builder."""
        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8000"

        builder = GenerateOutputsBuilder(
            env_id="test",
            env_args={},
            model="test-model",
            client=mock_client,
            state_columns=[],
            sampling_args={},
            results_path=Path("/tmp/test"),
        )

        existing = [
            RolloutOutput(
                example_id=0,
                task="test",
                prompt=[],
                completion=[],
                reward=1.0,
                timing={},
                is_completed=True,
                is_truncated=False,
                metrics={},
                rollout_idx=0,
            ),
            RolloutOutput(
                example_id=1,
                task="test",
                prompt=[],
                completion=[],
                reward=0.5,
                timing={},
                is_completed=True,
                is_truncated=False,
                metrics={},
                rollout_idx=0,
            ),
        ]

        builder.load_existing_outputs(existing)

        assert len(builder.outputs) == 2
        assert builder._existing_count == 2
        assert builder.get_new_outputs() == []
        assert builder.get_new_outputs_count() == 0

    def test_add_states_after_resume(self, make_state, make_output):
        """Test adding new states after loading existing outputs."""
        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8000"

        builder = GenerateOutputsBuilder(
            env_id="test",
            env_args={},
            model="test-model",
            client=mock_client,
            state_columns=[],
            sampling_args={},
            results_path=Path("/tmp/test"),
        )

        existing = [
            RolloutOutput(
                example_id=0,
                task="test",
                prompt=[],
                completion=[],
                reward=1.0,
                timing={},
                is_completed=True,
                is_truncated=False,
                metrics={},
                rollout_idx=0,
            ),
        ]
        builder.load_existing_outputs(existing)

        # Add new state
        new_state = make_state(example_id=1, reward=0.5, rollout_idx=0)
        builder.add_states([new_state])

        assert len(builder.outputs) == 2
        assert builder._existing_count == 1
        assert len(builder.get_new_outputs()) == 1
        assert builder.get_new_outputs_count() == 1
        assert builder.get_new_outputs()[0]["example_id"] == 1

    def test_cannot_load_after_adding_states(self, make_state):
        """Test that loading existing outputs after adding states raises error."""
        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8000"

        builder = GenerateOutputsBuilder(
            env_id="test",
            env_args={},
            model="test-model",
            client=mock_client,
            state_columns=[],
            sampling_args={},
            results_path=Path("/tmp/test"),
        )

        # Add a state first
        state = make_state(example_id=0, rollout_idx=0)
        builder.add_states([state])

        # Try to load existing outputs - should fail
        existing = [RolloutOutput(example_id=1, rollout_idx=0)]
        with pytest.raises(ValueError, match="Cannot load existing outputs"):
            builder.load_existing_outputs(existing)


class TestFindResumableRun:
    def test_find_by_direct_path(self, tmp_path):
        """Test finding run by direct path."""
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        (run_dir / "results.jsonl").touch()

        result = find_resumable_run(
            env_id="test",
            model="gpt-4",
            resume_arg=str(run_dir),
            env_dir_path=str(tmp_path / "environments"),
        )

        assert result == run_dir

    def test_direct_path_no_results_file(self, tmp_path):
        """Test that direct path without results.jsonl returns None."""
        run_dir = tmp_path / "my_run"
        run_dir.mkdir()
        # No results.jsonl file

        result = find_resumable_run(
            env_id="test",
            model="gpt-4",
            resume_arg=str(run_dir),
            env_dir_path=str(tmp_path / "environments"),
        )

        assert result is None

    def test_find_latest(self, tmp_path):
        """Test finding the latest run."""
        # Use unique identifiers to avoid conflicts with existing directories
        unique_id = tmp_path.name
        evals_dir = (
            tmp_path / "outputs" / "evals" / f"test-{unique_id}--gpt-4-{unique_id}"
        )
        evals_dir.mkdir(parents=True)

        # Create two runs
        run1 = evals_dir / "abc12345"
        run1.mkdir()
        (run1 / "results.jsonl").touch()

        run2 = evals_dir / "def67890"
        run2.mkdir()
        (run2 / "results.jsonl").touch()

        # Make run2 newer by touching it
        import time

        time.sleep(0.01)
        (run2 / "results.jsonl").touch()

        # Change to tmp_path to simulate working directory
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = find_resumable_run(
                env_id=f"test-{unique_id}",
                model=f"gpt-4-{unique_id}",
                resume_arg="latest",
                env_dir_path=str(tmp_path / "environments"),
            )
        finally:
            os.chdir(old_cwd)

        assert result is not None
        assert result.resolve() == run2.resolve()

    def test_find_by_uuid_prefix(self, tmp_path):
        """Test finding run by UUID prefix."""
        # Use unique identifiers to avoid conflicts
        unique_id = tmp_path.name
        evals_dir = (
            tmp_path / "outputs" / "evals" / f"test-{unique_id}--gpt-4-{unique_id}"
        )
        evals_dir.mkdir(parents=True)

        # Create run with unique prefix
        run = evals_dir / "xyz98765"
        run.mkdir()
        (run / "results.jsonl").touch()

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = find_resumable_run(
                env_id=f"test-{unique_id}",
                model=f"gpt-4-{unique_id}",
                resume_arg="xyz9",
                env_dir_path=str(tmp_path / "environments"),
            )
        finally:
            os.chdir(old_cwd)

        assert result is not None
        assert result.resolve() == run.resolve()

    def test_no_runs_found(self, tmp_path):
        """Test when no runs exist."""
        import os

        old_cwd = os.getcwd()
        unique_id = tmp_path.name
        try:
            os.chdir(tmp_path)
            result = find_resumable_run(
                env_id=f"test-{unique_id}",
                model=f"gpt-4-{unique_id}",
                resume_arg="latest",
                env_dir_path=str(tmp_path / "environments"),
            )
        finally:
            os.chdir(old_cwd)

        assert result is None


class TestFindEvalRuns:
    def test_find_runs_in_global_outputs(self, tmp_path):
        """Test finding runs in global outputs directory."""
        # Use unique identifiers to avoid conflicts
        unique_id = tmp_path.name
        evals_dir = (
            tmp_path / "outputs" / "evals" / f"test-{unique_id}--gpt-4-{unique_id}"
        )
        evals_dir.mkdir(parents=True)

        # Create runs
        run1 = evals_dir / "abc12345"
        run1.mkdir()
        (run1 / "results.jsonl").touch()

        run2 = evals_dir / "def67890"
        run2.mkdir()
        (run2 / "results.jsonl").touch()

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            runs = find_eval_runs(
                env_id=f"test-{unique_id}",
                model=f"gpt-4-{unique_id}",
                env_dir_path=str(tmp_path / "environments"),
            )
        finally:
            os.chdir(old_cwd)

        assert len(runs) == 2
        resolved_runs = [r.resolve() for r in runs]
        assert run1.resolve() in resolved_runs
        assert run2.resolve() in resolved_runs

    def test_find_runs_in_local_env(self, tmp_path):
        """Test finding runs in local environment outputs."""
        # Use unique identifiers to avoid conflicts
        unique_id = tmp_path.name
        env_dir = tmp_path / "environments" / f"test_{unique_id}"
        evals_dir = (
            env_dir / "outputs" / "evals" / f"test-{unique_id}--gpt-4-{unique_id}"
        )
        evals_dir.mkdir(parents=True)

        # Create run
        run = evals_dir / "abc12345"
        run.mkdir()
        (run / "results.jsonl").touch()

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            runs = find_eval_runs(
                env_id=f"test-{unique_id}",
                model=f"gpt-4-{unique_id}",
                env_dir_path=str(tmp_path / "environments"),
            )
        finally:
            os.chdir(old_cwd)

        assert len(runs) == 1
        assert runs[0].resolve() == run.resolve()

    def test_skip_dirs_without_results_file(self, tmp_path):
        """Test that directories without results.jsonl are skipped."""
        # Use unique identifiers to avoid conflicts
        unique_id = tmp_path.name
        evals_dir = (
            tmp_path / "outputs" / "evals" / f"test-{unique_id}--gpt-4-{unique_id}"
        )
        evals_dir.mkdir(parents=True)

        # Create run without results.jsonl
        run1 = evals_dir / "abc12345"
        run1.mkdir()

        # Create run with results.jsonl
        run2 = evals_dir / "def67890"
        run2.mkdir()
        (run2 / "results.jsonl").touch()

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            runs = find_eval_runs(
                env_id=f"test-{unique_id}",
                model=f"gpt-4-{unique_id}",
                env_dir_path=str(tmp_path / "environments"),
            )
        finally:
            os.chdir(old_cwd)

        assert len(runs) == 1
        resolved_runs = [r.resolve() for r in runs]
        assert run2.resolve() in resolved_runs
        assert run1.resolve() not in resolved_runs
