# Emulator GPT-5.4 Mini Eval Report

Generated on 2026-05-15 for branch `feat/emulator-benchmark`.

## Scope

This report covers three paths:

1. `configs/eval/emulator-gpt-5-4-mini-prime-smoke.toml` runs all ten emulator envs in `prime_smoke` mode. This validates environment loading, Prime sandbox provisioning, MiniSWEAgent command execution, and reward plumbing only.
2. `configs/eval/emulator-gpt-5-4-mini-suite.toml` runs the real implementation prompts and suite-backed verifier contracts for all ten emulator envs under `openai/gpt-5.4-mini`.
3. `configs/rl/emulator-chip8-prime-rl-smoke.toml` is the self-managed `prime-rl` bring-up config used on the provided 2 x RTX PRO 6000 node.

## Commands

Install local env packages before running these configs:

```bash
for env in \
  emulator_chip8 emulator_i8080_space_invaders emulator_gameboy_dmg emulator_nes \
  emulator_sms emulator_gameboy_cgb emulator_gba emulator_genesis emulator_snes \
  emulator_ps1
do
  uv pip install -e "environments/$env"
done
```

If the default public toolchain image is not pullable from the current Prime
account, set `PRIME_TOOLCHAIN_IMAGE` to a completed `programbench-toolchain`
image display ref from your image list before running `vf-eval`:

```bash
uv run prime images list --plain --output table
export PRIME_TOOLCHAIN_IMAGE="<displayRef for programbench-toolchain:latest>"
```

Run the fast all-env smoke:

```bash
uv run vf-eval configs/eval/emulator-gpt-5-4-mini-prime-smoke.toml --disable-tui
```

Run the real all-env suite:

```bash
uv run vf-eval configs/eval/emulator-gpt-5-4-mini-suite.toml --disable-tui
```

Run the self-managed training smoke after `prime-rl` is set up:

```bash
rl @ configs/rl/emulator-chip8-prime-rl-smoke.toml --output-dir outputs/emulator-chip8-smoke --clean-output-dir
```

## Prime Smoke Results

Source artifacts: `/tmp/emulator-gpt54-prime-smoke/evals/*/metadata.json` and `results.jsonl`.

The all-env smoke returned reward `1.00` for all ten environments. Treat this only as a sandbox and harness plumbing check. It does not run emulator ROM suites and must not be interpreted as GPT-5.4 Mini passing emulator correctness tests.

## Full Suite Results

Source artifacts: `/tmp/emulator-gpt54-suite/evals/*/*/metadata.json`.

| Env | Reward | Error | Public ROM pass | Build | Runner contract | Determinism | Turns | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `emulator-chip8` | 0.54 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 7 | 135s |
| `emulator-gameboy-cgb` | 0.54 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 8 | 151s |
| `emulator-gameboy-dmg` | 0.54 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 13 | 166s |
| `emulator-gba` | 0.54 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 7 | 122s |
| `emulator-genesis` | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 12 | 157s |
| `emulator-i8080-space-invaders` | 0.54 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 7 | 140s |
| `emulator-nes` | 0.50 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 10 | 159s |
| `emulator-ps1` | 0.54 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 9 | 167s |
| `emulator-sms` | 0.54 | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 | 6 | 87s |
| `emulator-snes` | 0.52 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 9 | 146s |

Aggregate suite status:

```text
envs evaluated       : 10
mean reward          : 0.58
mean error           : 0.10
full public pass envs: 1 / 10
public pass sum      : 1.00
error envs           : emulator-sms
```

Interpretation:

- `openai/gpt-5.4-mini` did not pass all emulator tests. Nine of ten envs had `public_rom_pass_rate = 0.00`.
- The suite shows that the build path, runner contract, and verifier scoring path execute across the full curriculum.
- The Genesis result reached `public_rom_pass_rate = 1.00` against the currently configured public source manifest. Keep private artifacts mounted before treating that as comprehensive Genesis coverage.
- The SMS rollout recorded `avg_error = 1.00` and `emulator_agent_exit_status = -1.00`; this is a rollout failure signal, not a solved emulator.

## Training Configs

Hosted Training config: `configs/rl/emulator-chip8-minimal.toml`.

The hosted example uses `openai/gpt-oss-20b`, a Hosted Training model, because `openai/gpt-5.4-mini` is used here as an evaluator rather than a fine-tuning target.

```bash
uv run prime train configs/rl/emulator-chip8-minimal.toml
```

Self-managed node config: `configs/rl/emulator-chip8-prime-rl-smoke.toml`.

This config targets a 2-GPU machine with one inference GPU and one trainer GPU. It uses `Qwen/Qwen3-0.6B` for a one-step smoke run, enables renderer mode for `prime-rl` trajectory metadata, and passes `inline_system_prompt = true` so the emulator guidance is a user message instead of a separate system message.

## Node Validation

Node: `root@31.22.104.54`, hostname `chess-dagger-2xpro6000b`, 2 x NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB.

Validated setup:

- Branch cloned on the node at `/root/verifiers-emulator`.
- `uv sync` completed.
- All ten emulator env packages installed editable into the repo venv.
- `uv run pytest tests/test_emulator_benchmark_envs.py -q` passed with `53 passed`.
- `prime lab setup --prime-rl --skip-agents-md --skip-install --no-interactive` completed.
- The current repo and all ten emulator env packages were installed editable into `prime-rl/.venv`.
- `renderers==0.1.8.dev0` was installed into `prime-rl/.venv` so the self-managed run uses the renderer API expected by this branch.
- `rl @ prime-rl/configs/emulator_chip8_smoke.toml --dry-run` completed successfully with renderer mode.

Completed training smoke:

```text
output dir   : /root/verifiers-emulator/outputs/emulator-chip8-smoke
model        : Qwen/Qwen3-0.6B
orchestrator : Step 0, reward 0.5400, 3868 tokens/sample, 105.12s
trainer      : Step 0, loss 0.0000, entropy 0.5280, mismatch KL 0.0019
trainer      : grad norm 0.0019, LR 1.00e-06, throughput 9202 tokens/s
checkpoint   : outputs/emulator-chip8-smoke/checkpoints/step_1/trainer/
weights      : outputs/emulator-chip8-smoke/weights/step_1/
```

The training smoke required `verifiers.v1.Env.apply_controls()` to mirror `sampling_args` onto the top-level rollout state. `prime-rl` reads `sampling_args` from saved rollout outputs while v1 runtime controls already store the same value under `state["runtime"]["sampling_args"]`.

CPU build node validation:

```text
node       : root@86.38.238.176
hostname   : pb-coordinator
branch     : feat/emulator-benchmark
command    : uv run pytest tests/test_emulator_benchmark_envs.py tests/test_v1_runtime_lifecycle.py::test_v1_env_apply_controls_mirrors_sampling_args_for_rollout_outputs -q
result     : 56 passed
```

## Private Eval Notes

- Some public upstream repositories are source-only and do not ship ready ROMs. The verifier supports mounted/generated private artifacts through `EMULATOR_PRIVATE_ARTIFACT_DIR`.
- Do not commit copyrighted ROM or BIOS bytes. Mount them privately and require SHA-256 manifests before scoring.
- Start training with CHIP-8 until reward diversity is visible, then expand to `emulator-nes` and `emulator-gameboy-dmg`. Keep the harder systems eval-only until the lower-level tasks show nonzero public ROM pass rates.
