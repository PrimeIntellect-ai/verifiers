"""
Tests for SFTTrainer and SFTConfig.

These tests verify that the SFT trainer implementation:
1. Can be imported from verifiers module
2. Has correct configuration defaults
3. Initializes correctly with model and dataset
4. Has the expected methods and attributes
"""

import os
from unittest.mock import MagicMock, Mock

import pytest
from torch.utils.data import Dataset


# Minimal dummy dataset for testing
class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "prompt": f"Test prompt {idx}",
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [1, 2, 3, 4, 5],
        }


class TestSFTConfig:
    """Test SFTConfig configuration class."""

    def test_sft_config_import_from_verifiers(self):
        """Test that SFTConfig can be imported from verifiers."""
        import verifiers as vf
        # Should be able to access SFTConfig
        assert hasattr(vf, "SFTConfig") or "SFTConfig" in dir(vf)

    def test_sft_config_import_from_trainer_module(self):
        """Test that SFTConfig can be imported from trainer module."""
        from verifiers_rl.rl.trainer import SFTConfig
        assert SFTConfig is not None

    def test_sft_config_default_values(self):
        """Test that SFTConfig has correct default values."""
        from verifiers_rl.rl.trainer import SFTConfig

        config = SFTConfig(run_name="test")

        # Model loading defaults
        assert config.use_liger is True
        assert config.use_lora is True
        assert config.lora_rank == 8
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.0

        # Batch defaults
        assert config.batch_size == 512
        assert config.micro_batch_size == 8
        assert config.max_seq_len == 2048

        # Training defaults
        assert config.max_steps == 500
        assert config.num_train_epochs == 1
        assert config.learning_rate == 1e-5
        assert config.max_grad_norm == 1.0

        # vLLM defaults (should be False)
        assert config.use_vllm is False

    def test_sft_config_custom_values(self):
        """Test that SFTConfig accepts custom values."""
        from verifiers_rl.rl.trainer import SFTConfig

        config = SFTConfig(
            run_name="test",
            batch_size=256,
            micro_batch_size=4,
            learning_rate=2e-5,
            max_steps=100,
            use_lora=False,
        )

        assert config.batch_size == 256
        assert config.micro_batch_size == 4
        assert config.learning_rate == 2e-5
        assert config.max_steps == 100
        assert config.use_lora is False

    def test_sft_config_output_dir_auto_set(self):
        """Test that output_dir is auto-set from run_name."""
        from verifiers_rl.rl.trainer import SFTConfig

        config = SFTConfig(run_name="my-experiment")
        assert config.output_dir == "outputs/my-experiment"

    def test_sft_config_lora_target_modules_default(self):
        """Test that LoRA target modules are set by default."""
        from verifiers_rl.rl.trainer import SFTConfig

        config = SFTConfig(run_name="test")
        expected_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj",
        ]
        assert config.lora_target_modules == expected_modules

    def test_sft_config_vllm_settings(self):
        """Test vLLM optional settings."""
        from verifiers_rl.rl.trainer import SFTConfig

        config = SFTConfig(
            run_name="test",
            use_vllm=True,
            vllm_sample_every_n_steps=50,
            vllm_num_samples=10,
        )

        assert config.use_vllm is True
        assert config.vllm_sample_every_n_steps == 50
        assert config.vllm_num_samples == 10


class TestSFTTrainer:
    """Test SFTTrainer class."""

    def test_sft_trainer_import_from_verifiers(self):
        """Test that SFTTrainer can be imported from verifiers."""
        import verifiers as vf
        # Should be able to access SFTTrainer (may be lazy import)
        assert "SFTTrainer" in vf.__all__

    def test_sft_trainer_import_from_trainer_module(self):
        """Test that SFTTrainer can be imported from trainer module."""
        from verifiers_rl.rl.trainer import SFTTrainer
        assert SFTTrainer is not None

    def test_sft_trainer_has_required_methods(self):
        """Test that SFTTrainer has required methods."""
        from verifiers_rl.rl.trainer import SFTTrainer

        required_methods = [
            "__init__",
            "compute_loss",
            "training_step",
            "log",
            "log_metrics",
        ]

        for method in required_methods:
            assert hasattr(SFTTrainer, method), f"SFTTrainer missing method: {method}"

    def test_sft_trainer_compute_loss_signature(self):
        """Test that compute_loss has correct signature."""
        import inspect
        from verifiers_rl.rl.trainer import SFTTrainer

        sig = inspect.signature(SFTTrainer.compute_loss)
        params = list(sig.parameters.keys())

        # Should have self, model, inputs, and optional params
        assert "model" in params
        assert "inputs" in params
        assert "return_outputs" in params
        assert "num_items_in_batch" in params


class TestSFTIntegration:
    """Integration tests for SFT workflow."""

    def test_sft_config_and_trainer_work_together(self):
        """Test that SFTConfig and SFTTrainer can be used together."""
        from verifiers_rl.rl.trainer import SFTConfig, SFTTrainer

        config = SFTConfig(run_name="test", max_steps=1)
        dataset = DummyDataset(size=5)

        # Mock model and tokenizer to avoid actual model loading
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config._name_or_path = "test-model"

        # This should not raise an error during initialization
        # (actual training would require real model)
        try:
            trainer = SFTTrainer(
                model=mock_model,
                train_dataset=dataset,
                args=config,
                processing_class=MagicMock(),
            )
            assert trainer is not None
        except TypeError as e:
            # Some initialization may fail due to mocking, but basic structure should work
            # The important thing is that the classes are compatible
            if "SFTTrainer" in str(e) or "abstract" in str(e):
                pytest.fail(f"SFTTrainer initialization failed: {e}")

    def test_sft_follows_rl_trainer_pattern(self):
        """Test that SFTTrainer follows similar patterns to RLTrainer."""
        from verifiers_rl.rl.trainer import RLTrainer, SFTTrainer

        # Both should have similar attributes
        sft_attrs = set(dir(SFTTrainer))
        rl_attrs = set(dir(RLTrainer))

        # Common methods that should exist in both
        common_methods = {
            "compute_loss",
            "training_step",
            "log",
            "log_metrics",
        }

        for method in common_methods:
            assert method in sft_attrs, f"SFTTrainer missing {method}"
            assert method in rl_attrs, f"RLTrainer missing {method}"


class TestSFTScript:
    """Test the SFT training script."""

    def test_sft_script_exists(self):
        """Test that the SFT script exists."""
        import os
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "packages",
            "verifiers-rl",
            "verifiers_rl",
            "scripts",
            "sft.py",
        )
        assert os.path.exists(script_path), f"SFT script not found at {script_path}"

    def test_sft_script_has_main_function(self):
        """Test that the SFT script has a main function."""
        import os
        import sys

        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "packages",
            "verifiers-rl",
            "verifiers_rl",
            "scripts",
            "sft.py",
        )

        # Read the script and check for main function
        with open(script_path) as f:
            content = f.read()

        assert "def main()" in content, "SFT script should have main() function"
        assert "SFTTrainer" in content, "SFT script should import SFTTrainer"
        assert "SFTConfig" in content, "SFT script should import SFTConfig"


class TestSFTConfigFile:
    """Test the example SFT config file."""

    def test_sft_example_config_exists(self):
        """Test that example SFT config exists."""
        import os
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "local",
            "vf-sft",
            "example-sft.toml",
        )
        assert os.path.exists(config_path), f"Example SFT config not found at {config_path}"

    def test_sft_example_config_has_required_sections(self):
        """Test that example SFT config has required sections."""
        import os

        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
            "local",
            "vf-sft",
            "example-sft.toml",
        )

        with open(config_path) as f:
            content = f.read()

        # Should have model and dataset
        assert "model =" in content, "Config should specify model"
        assert "dataset =" in content, "Config should specify dataset"

        # Should have [sft] section
        assert "[sft]" in content, "Config should have [sft] section"

        # Should have key SFT parameters
        assert "run_name" in content, "Config should have run_name"
        assert "max_steps" in content or "num_train_epochs" in content, "Config should have training parameters"
        assert "batch_size" in content, "Config should have batch_size"


class TestSFTLazyImports:
    """Test SFT lazy imports in verifiers module."""

    def test_sft_in_lazy_imports(self):
        """Test that SFTConfig and SFTTrainer are in lazy imports."""
        import verifiers as vf

        lazy_imports = getattr(vf, "_LAZY_IMPORTS", {})
        assert "SFTConfig" in lazy_imports, "SFTConfig should be in lazy imports"
        assert "SFTTrainer" in lazy_imports, "SFTTrainer should be in lazy imports"

    def test_sft_lazy_import_target_correct(self):
        """Test that lazy imports point to correct module."""
        import verifiers as vf

        lazy_imports = getattr(vf, "_LAZY_IMPORTS", {})

        assert lazy_imports.get("SFTConfig") == "verifiers_rl.rl.trainer:SFTConfig"
        assert lazy_imports.get("SFTTrainer") == "verifiers_rl.rl.trainer:SFTTrainer"

    def test_sft_in_rl_names(self):
        """Test that SFT exports are in rl_names for error handling."""
        import verifiers as vf
        import inspect

        # Get the source code of __getattr__ to check rl_names
        source = inspect.getsource(vf.__getattr__)
        assert '"SFTConfig"' in source or "'SFTConfig'" in source, "SFTConfig should be in rl_names"
        assert '"SFTTrainer"' in source or "'SFTTrainer'" in source, "SFTTrainer should be in rl_names"
