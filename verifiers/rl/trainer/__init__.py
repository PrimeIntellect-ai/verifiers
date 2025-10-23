import logging

import torch._dynamo

from .config import RLConfig
from .trainer import RLTrainer

torch._dynamo.config.suppress_errors = True


def GRPOTrainer(model, processing_class, env, args):
    logging.warning("GRPOTrainer is deprecated and renamed to RLTrainer.")
    return RLTrainer(model, processing_class, env, args)


def GRPOConfig(**kwargs):
    logging.warning("GRPOConfig is deprecated and renamed to RLConfig.")
    return RLConfig(**kwargs)


def grpo_defaults(**kwargs):
    logging.warning("grpo_defaults is deprecated and replaced with RLConfig.")
    return RLConfig(**kwargs)


def lora_defaults(**kwargs):
    raise ValueError("lora_defaults is deprecated and replaced with RLConfig.")


__all__ = [
    "RLConfig",
    "RLTrainer",
    "GRPOTrainer",
    "GRPOConfig",
    "grpo_defaults",
    "lora_defaults",
]
