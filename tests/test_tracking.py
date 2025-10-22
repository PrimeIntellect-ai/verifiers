# ABOUTME: Tests for the tracking module including Tracker base class,
# ABOUTME: NullTracker, CompositeTracker, CSVTracker, WandbTracker, MLFlowTracker, and TensorBoardTracker.

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from verifiers.tracking import (
    CSVTracker,
    CompositeTracker,
    MLFlowTracker,
    NullTracker,
    TensorBoardTracker,
    Tracker,
    WandbTracker,
)


class TestNullTracker:
    """Test cases for NullTracker."""

    def test_null_tracker_initialization(self):
        """Test NullTracker initialization."""
        tracker = NullTracker()
        assert tracker.project is None
        assert tracker.name is None
        assert tracker.config == {}
        assert not tracker._initialized

    def test_null_tracker_init(self):
        """Test NullTracker init method."""
        tracker = NullTracker()
        tracker.init()
        assert tracker._initialized

    def test_null_tracker_log_metrics(self):
        """Test NullTracker log_metrics does nothing."""
        tracker = NullTracker()
        tracker.log_metrics({"accuracy": 0.95, "loss": 0.05}, step=1)

    def test_null_tracker_log_table(self):
        """Test NullTracker log_table does nothing."""
        tracker = NullTracker()
        table_data = {
            "prompt": ["test1", "test2"],
            "completion": ["answer1", "answer2"],
        }
        tracker.log_table("test_table", table_data, step=1)

    def test_null_tracker_log_completions(self):
        """Test NullTracker log_completions does nothing."""
        tracker = NullTracker()
        tracker.log_completions(["prompt"], ["completion"], [0.9], step=1)

    def test_null_tracker_log_config(self):
        """Test NullTracker log_config updates config."""
        tracker = NullTracker()
        config = {"learning_rate": 0.001}
        tracker.log_config(config)
        assert tracker.config == config

    def test_null_tracker_finish(self):
        """Test NullTracker finish method."""
        tracker = NullTracker()
        tracker.finish()


class TestCompositeTracker:
    """Test cases for CompositeTracker."""

    def test_composite_tracker_initialization_empty(self):
        """Test CompositeTracker initialization with no trackers."""
        tracker = CompositeTracker([])
        assert tracker.trackers == []

    def test_composite_tracker_initialization_with_trackers(self):
        """Test CompositeTracker initialization with multiple trackers."""
        t1 = NullTracker()
        t2 = NullTracker()
        tracker = CompositeTracker([t1, t2])
        assert len(tracker.trackers) == 2
        assert tracker.trackers[0] is t1
        assert tracker.trackers[1] is t2

    def test_composite_tracker_init(self):
        """Test CompositeTracker init calls all sub-trackers."""
        t1 = NullTracker()
        t2 = NullTracker()
        tracker = CompositeTracker([t1, t2])

        tracker.init()

        assert t1._initialized
        assert t2._initialized

    def test_composite_tracker_log_metrics(self):
        """Test CompositeTracker log_metrics calls all sub-trackers."""
        t1 = MagicMock(spec=Tracker)
        t2 = MagicMock(spec=Tracker)
        tracker = CompositeTracker([t1, t2])

        metrics = {"accuracy": 0.95}
        tracker.log_metrics(metrics, step=1)

        t1.log_metrics.assert_called_once_with(metrics, step=1)
        t2.log_metrics.assert_called_once_with(metrics, step=1)

    def test_composite_tracker_log_table(self):
        """Test CompositeTracker log_table calls all sub-trackers."""
        t1 = MagicMock(spec=Tracker)
        t2 = MagicMock(spec=Tracker)
        tracker = CompositeTracker([t1, t2])

        table_data = {"col1": [1, 2], "col2": [3, 4]}
        tracker.log_table("test", table_data, step=1)

        t1.log_table.assert_called_once_with("test", table_data, step=1)
        t2.log_table.assert_called_once_with("test", table_data, step=1)

    def test_composite_tracker_log_completions(self):
        """Test CompositeTracker log_completions calls all sub-trackers."""
        t1 = MagicMock(spec=Tracker)
        t2 = MagicMock(spec=Tracker)
        tracker = CompositeTracker([t1, t2])

        prompts = ["p1"]
        completions = ["c1"]
        rewards = [0.9]
        tracker.log_completions(prompts, completions, rewards, step=1)

        t1.log_completions.assert_called_once_with(prompts, completions, rewards, step=1)
        t2.log_completions.assert_called_once_with(prompts, completions, rewards, step=1)

    def test_composite_tracker_log_config(self):
        """Test CompositeTracker log_config calls all sub-trackers."""
        t1 = MagicMock(spec=Tracker)
        t2 = MagicMock(spec=Tracker)
        tracker = CompositeTracker([t1, t2])

        config = {"lr": 0.001}
        tracker.log_config(config)

        t1.log_config.assert_called_once_with(config)
        t2.log_config.assert_called_once_with(config)

    def test_composite_tracker_finish(self):
        """Test CompositeTracker finish calls all sub-trackers."""
        t1 = MagicMock(spec=Tracker)
        t2 = MagicMock(spec=Tracker)
        tracker = CompositeTracker([t1, t2])

        tracker.finish()

        t1.finish.assert_called_once()
        t2.finish.assert_called_once()


class TestCSVTracker:
    """Test cases for CSVTracker."""

    def test_csv_tracker_initialization(self, tmp_path):
        """Test CSVTracker initialization."""
        log_dir = str(tmp_path / "logs")
        tracker = CSVTracker(log_dir=log_dir, project="test", name="exp1")

        assert tracker.log_dir == Path(log_dir)
        assert tracker.project == "test"
        assert tracker.name == "exp1"

    def test_csv_tracker_init_creates_directory(self, tmp_path):
        """Test CSVTracker init creates log directory."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))

        assert not log_dir.exists()
        tracker.init()
        assert log_dir.exists()

    def test_csv_tracker_init_saves_config(self, tmp_path):
        """Test CSVTracker init saves config to JSON."""
        log_dir = tmp_path / "logs"
        config = {"learning_rate": 0.001, "batch_size": 32}
        tracker = CSVTracker(log_dir=str(log_dir), config=config)

        tracker.init()

        config_file = log_dir / "config.json"
        assert config_file.exists()

        with open(config_file) as f:
            saved_config = json.load(f)
        assert saved_config == config

    def test_csv_tracker_log_metrics(self, tmp_path):
        """Test CSVTracker log_metrics writes to CSV."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))
        tracker.init()

        tracker.log_metrics({"accuracy": 0.95, "loss": 0.05}, step=1)
        tracker.log_metrics({"accuracy": 0.96, "loss": 0.04}, step=2)

        metrics_file = log_dir / "metrics.csv"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert "step,accuracy,loss" in lines[0]
        assert "1,0.95,0.05" in lines[1]
        assert "2,0.96,0.04" in lines[2]

    def test_csv_tracker_log_metrics_no_step(self, tmp_path):
        """Test CSVTracker log_metrics without step."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))
        tracker.init()

        tracker.log_metrics({"accuracy": 0.95})

        metrics_file = log_dir / "metrics.csv"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert "accuracy" in lines[0]

    def test_csv_tracker_log_metrics_dynamic_columns(self, tmp_path):
        """Test CSVTracker handles dynamic columns in metrics."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))
        tracker.init()

        tracker.log_metrics({"accuracy": 0.95}, step=1)
        tracker.log_metrics({"accuracy": 0.96, "loss": 0.04}, step=2)

        metrics_file = log_dir / "metrics.csv"
        with open(metrics_file) as f:
            lines = f.readlines()

        assert "step,accuracy" in lines[0]

    def test_csv_tracker_log_table(self, tmp_path):
        """Test CSVTracker log_table writes table to CSV."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))
        tracker.init()

        table_data = {
            "prompt": ["p1", "p2"],
            "completion": ["c1", "c2"],
            "reward": [0.9, 0.8],
        }
        tracker.log_table("completions", table_data)

        table_file = log_dir / "completions.csv"
        assert table_file.exists()

        with open(table_file) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert "prompt,completion,reward" in lines[0]

    def test_csv_tracker_log_table_empty_data(self, tmp_path):
        """Test CSVTracker handles empty table data."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))
        tracker.init()

        tracker.log_table("empty", {})

        table_file = log_dir / "empty.csv"
        assert not table_file.exists()

    def test_csv_tracker_log_table_inconsistent_lengths(self, tmp_path):
        """Test CSVTracker handles inconsistent column lengths."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))
        tracker.init()

        table_data = {
            "col1": [1, 2, 3],
            "col2": [4, 5],
        }
        tracker.log_table("inconsistent", table_data)

        table_file = log_dir / "inconsistent.csv"
        assert not table_file.exists()

    def test_csv_tracker_log_table_with_complex_values(self, tmp_path):
        """Test CSVTracker handles complex values in tables."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))
        tracker.init()

        table_data = {
            "prompt": ["p1"],
            "metadata": [{"key": "value"}],
            "tags": [["tag1", "tag2"]],
        }
        tracker.log_table("complex", table_data)

        table_file = log_dir / "complex.csv"
        assert table_file.exists()

        with open(table_file) as f:
            lines = f.readlines()

        assert '"{""key"": ""value""}"' in lines[1] or '"{\\"key\\": \\"value\\"}"' in lines[1]

    def test_csv_tracker_log_config(self, tmp_path):
        """Test CSVTracker log_config updates config file."""
        log_dir = tmp_path / "logs"
        tracker = CSVTracker(log_dir=str(log_dir))
        tracker.init()

        new_config = {"updated": True}
        tracker.log_config(new_config)

        config_file = log_dir / "config.json"
        with open(config_file) as f:
            saved_config = json.load(f)

        assert saved_config == new_config


class TestWandbTracker:
    """Test cases for WandbTracker."""

    def test_wandb_tracker_initialization(self):
        """Test WandbTracker initialization."""
        tracker = WandbTracker(
            project="test_project", name="test_run", entity="test_entity"
        )

        assert tracker.project == "test_project"
        assert tracker.name == "test_run"
        assert tracker.entity == "test_entity"
        assert tracker._wandb is None
        assert tracker._run is None

    @pytest.mark.skip(reason="Requires wandb to be installed")
    def test_wandb_tracker_integration(self):
        """Integration test - requires wandb installation."""
        pass


class TestMLFlowTracker:
    """Test cases for MLFlowTracker."""

    def test_mlflow_tracker_initialization(self):
        """Test MLFlowTracker initialization."""
        tracker = MLFlowTracker(
            experiment_name="test_experiment",
            run_name="test_run",
            tracking_uri="http://localhost:5000",
        )

        assert tracker.experiment_name == "test_experiment"
        assert tracker.run_name == "test_run"
        assert tracker.tracking_uri == "http://localhost:5000"
        assert tracker._mlflow is None
        assert tracker._run is None

    @pytest.mark.skip(reason="Requires mlflow to be installed")
    def test_mlflow_tracker_integration(self):
        """Integration test - requires mlflow installation."""
        pass


class TestTensorBoardTracker:
    """Test cases for TensorBoardTracker."""

    def test_tensorboard_tracker_initialization(self):
        """Test TensorBoardTracker initialization."""
        tracker = TensorBoardTracker(log_dir="./runs", comment="test")

        assert tracker.log_dir == "./runs"
        assert tracker.comment == "test"
        assert tracker._writer is None
        assert not tracker._hparams_logged

    @pytest.mark.skip(reason="Requires torch to be installed")
    def test_tensorboard_tracker_integration(self):
        """Integration test - requires torch installation."""
        pass
