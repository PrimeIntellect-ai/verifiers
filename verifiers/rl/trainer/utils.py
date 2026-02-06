from importlib import import_module

_mod = import_module("verifiers_rl.rl.trainer.utils")
get_model = _mod.get_model
get_model_and_tokenizer = _mod.get_model_and_tokenizer
