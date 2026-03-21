# Template for training VeriHop with Prime RL.
#
# 1. Install this env: from repo root, `uv pip install -e environments/verihop`
# 2. In your Prime RL TOML, register an orchestrator env whose `name` resolves to
#    `verihop` (or your fork) and pass kwargs such as:
#    args = { num_samples = 10000, seed = 42, use_tools = false }
#
# Example (illustrative only — field names follow your prime-rl version):
#
# [[orchestrator.env]]
# id = "verihop-train"
# name = "verihop"
# args = { num_samples = 8192, process_weight = 0.35, outcome_weight = 0.65 }

from verihop import load_environment

if __name__ == "__main__":
    env = load_environment(num_samples=16, seed=1, use_tools=False)
    print(env)
