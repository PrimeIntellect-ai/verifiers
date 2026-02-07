from importlib import import_module

_mod = import_module("verifiers_rl.rl.trainer.trainer")
RLTrainer = _mod.RLTrainer
