from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonData: TypeAlias = dict[str, JsonValue]

CONTRACT_VERSION = "emulator-runner-v1"
DEFAULT_EXECUTABLE = "/workspace/target/release/emulator-runner"
DEFAULT_JSON_OUTPUT = "/tmp/emulator_case_output.json"


def runner_contract() -> JsonData:
    return {
        "version": CONTRACT_VERSION,
        "build_command": "cargo build --release",
        "executable_path": DEFAULT_EXECUTABLE,
        "self_test": [DEFAULT_EXECUTABLE, "--self-test"],
        "contract_version": [DEFAULT_EXECUTABLE, "--contract-version"],
        "case_command": [
            DEFAULT_EXECUTABLE,
            "--platform",
            "<platform>",
            "--case",
            "<case_id>",
            "--rom",
            "<rom_or_suite_artifact_path>",
            "--cycles",
            "<cycle_budget>",
            "--frames",
            "<frame_budget>",
            "--output",
            "<json_output_path>",
        ],
        "case_output_schema": {
            "required": ["platform", "case_id", "passed"],
            "optional": [
                "cycles",
                "frames",
                "serial",
                "framebuffer_sha256",
                "frame_sequence_sha256",
                "trace_crc32",
                "audio_crc32",
                "perf",
                "log",
            ],
        },
    }


def runner_contract_markdown() -> str:
    return """\
## Emulator runner contract

The grader builds the Rust crate with:

```sh
cargo build --release
```

It then expects an executable at `target/release/emulator-runner`. The runner
must support these commands:

```sh
emulator-runner --contract-version
emulator-runner --self-test
emulator-runner --platform <slug> --case <case_id> --rom <path> \\
  --cycles <cycle_budget> --frames <frame_budget> --output <json_path>
```

`--rom` is the input artifact path. Most cases pass a ROM image; processor-test
suites can pass a machine-readable fixture such as a JSON or compressed JSON
file when the case kind documents that contract.

The per-case JSON output must include:

```json
{
  "platform": "chip8",
  "case_id": "chip8_timendus_suite",
  "passed": true,
  "cycles": 10000,
  "frames": 1,
  "framebuffer_sha256": "...",
  "frame_sequence_sha256": "...",
  "trace_crc32": "...",
  "audio_crc32": "...",
  "serial": "optional pass/fail text",
  "perf": {"fps": 60.0}
}
```

For CPU/trace tests, emit `trace_crc32` or pass/fail serial text. For visual
tests, emit `framebuffer_sha256` or `frame_sequence_sha256`. For audio-capable
platforms, emit `audio_crc32` when the test exercises audio. Re-running the same
ROM/case/cycle budget must produce identical JSON signals.
"""
