# Emulator Prime Env Training-Readiness Audit

Generated on 2026-05-14 for branch `feat/emulator-benchmark`.

## Objective Checklist

| Requirement | Evidence |
| --- | --- |
| Ten emulator Prime envs exist | `environments/emulator_{chip8,i8080_space_invaders,gameboy_dmg,nes,sms,gameboy_cgb,gba,genesis,snes,ps1}` |
| Shared suite verifier replaces scaffold-only scoring | `environments/emulator_common/suite_adapters.py` fetches sources, discovers public/private artifacts, invokes runner cases, scores build/contract/source/pass-rate/signals/determinism/runtime |
| Standard runner contract documented | `environments/emulator_common/runner_contract.py`; generated `README.md` and `verification/runner_contract.json` include the contract |
| Private artifact support | `EMULATOR_PRIVATE_ARTIFACT_DIR` or `/private/emulator`; mounted files/directories are included in executed artifacts |
| RL score breakdowns | `build_score`, `public_rom_pass_rate`, `framebuffer_hash_score`, `trace_cpu_score`, `audio_perf_score`, `deterministic_replay_score`, `runtime_stability_score`, `runner_contract_score` |
| No committed private ROM/BIOS bytes | Manifest-only private artifact declarations; secret scan over emulator envs/tests found no team ID, API key, private key, GitHub token, or HF token |
| Local tests | `uv run pytest tests/test_emulator_benchmark_envs.py` -> `47 passed`; `uv run ruff check ...` -> `All checks passed`; source verification all OK |
| CPU-node tests | On `root@86.38.238.176`: pytest `47 passed`, ruff OK, source verification all OK |
| Generated starter crate build gate | On CPU node: all ten generated Rust starters passed `cargo test`, `cargo build --release`, `--contract-version`, `--self-test`, and deterministic no-arg JSON smoke |
| Real verifier invocation | On CPU node: CHIP-8 starter fetched Timendus and executed 8 `.ch8` artifacts; reward `0.54`, public pass rate `0.0`, build/contract/source/stability all `1.0` |
| Real gpt-5.4-mini eval | `/tmp/emulator-real-suite-gpt54-priority/.../results.jsonl`: reward `0.54`; verifier metrics populated from the actual suite path |
| Prime smoke all ten | `/tmp/emulator-prime-smoke-after/evals/*/results.jsonl`: all ten reward `1.0`, stop `command_completed` |

## Env Mapping

| Env | Public suites wired | Executable public artifacts found | Private/generated artifacts expected | Local | CPU | Prime/gpt-5.4-mini |
| --- | --- | ---: | --- | --- | --- | --- |
| `emulator-chip8` | Timendus suite covering IBM/Corax/quirks | 8 | None | Pass | Pass | real eval reward `0.54`; prime smoke `1.0` |
| `emulator-i8080-space-invaders` | PCjs 8080 exerciser source/page | 9 exerciser source fixtures | `space_invaders_rom_set` | Pass | Pass | prime smoke `1.0` |
| `emulator-gameboy-dmg` | c-sp index, mooneye, dmg-acid2, SameSuite | 0 prebuilt in fetched repos | `gameboy_dmg_public_test_roms`, `pokemon_red_rom` | Pass | Pass | prime smoke `1.0` |
| `emulator-nes` | christopherpow NES test ROM archive, NESdev wiki | 128 cap | None | Pass | Pass | prime smoke `1.0` |
| `emulator-sms` | SMSTestSuite, redcode Z80, SMS Power page | 0 prebuilt `.sms` ROMs | `sms_test_suite_roms`, `zexdoc_zexall_roms` | Pass | Pass | prime smoke `1.0` |
| `emulator-gameboy-cgb` | cgb-acid2, mooneye, SameSuite, c-sp index | 0 prebuilt in fetched repos | `gameboy_cgb_public_test_roms` | Pass | Pass | prime smoke `1.0` |
| `emulator-gba` | mGBA suite, jsmolka gba-tests, TONC, NBA hw-test | 38 | `gba_bios` | Pass | Pass | prime smoke `1.0` |
| `emulator-genesis` | Tom Harte 68000 sparse fixtures, redcode Z80, 240p, Sega Retro page | 128 cap, mainly Tom Harte `.json.gz` | `genesis_240p_suite_rom`, `genesis_z80_fixture_roms`, `sonic_rom` | Pass | Pass | prime smoke `1.0` |
| `emulator-snes` | Peter Lemon, undisbeliever, SNESdev, Snes Central/blargg | 128 cap | None | Pass | Pass | prime smoke `1.0` |
| `emulator-ps1` | Peter Lemon PSX, psxsdk, psx-spx GPU spec | 128 cap | `ps1_bios` | Pass | Pass | prime smoke `1.0` |

## Remaining Limitations

- Several public upstream repos are source-only and do not ship ready emulator ROMs. Those suites are intentionally represented as mounted/generated artifacts through the documented private artifact paths above.
- The verifier executes discovered or mounted artifacts through the standard runner contract; it does not commit or vendor copyrighted ROM/BIOS bytes.
- The gpt-5.4-mini real rollout still had an agent command error, but the reward path now runs the verifier anyway and records the agent error separately, producing suite-based reward components.
