import importlib


try:
    _trainer = importlib.import_module("verifiers_rl.rl.trainer")
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "verifiers.rl.trainer moved to optional package 'verifiers-rl'. Install with: uv add verifiers-rl"
    ) from e

RLTrainer = _trainer.RLTrainer
RLConfig = _trainer.RLConfig
GRPOTrainer = _trainer.GRPOTrainer
GRPOConfig = _trainer.GRPOConfig
grpo_defaults = _trainer.grpo_defaults
lora_defaults = _trainer.lora_defaults
get_model = _trainer.get_model
get_model_and_tokenizer = _trainer.get_model_and_tokenizer

__all__ = [
    "RLTrainer",
    "RLConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "grpo_defaults",
    "lora_defaults",
    "get_model",
    "get_model_and_tokenizer",
]
