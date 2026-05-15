import json
import textwrap

from verifiers.v1.types import ConfigData, ProgramData

from .runner_contract import runner_contract, runner_contract_markdown
from .task_schema import EmulatorManifest


def cargo_toml_file(task: ConfigData, state: ConfigData | None = None) -> str:
    _ = state
    package = str(task["info"]["slug"]).replace("_", "-")
    return f"""\
[package]
name = "{package}"
version = "0.1.0"
edition = "2021"

[lib]
name = "emulator_benchmark"
path = "src/lib.rs"

[[bin]]
name = "emulator-runner"
path = "src/main.rs"

[dependencies]
"""


def starter_lib_file(task: ConfigData, state: ConfigData | None = None) -> str:
    _ = state
    info = task["info"]
    width = int(info.get("framebuffer", {}).get("width", 160))
    height = int(info.get("framebuffer", {}).get("height", 144))
    cycles = int(info.get("runtime", {}).get("smoke_cycles", 1024))
    return f"""\
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmulatorError(pub String);

impl std::fmt::Display for EmulatorError {{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{
        f.write_str(&self.0)
    }}
}}

impl std::error::Error for EmulatorError {{}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {{
    pub cycles: u64,
    pub framebuffer: Vec<u8>,
}}

#[derive(Debug, Clone)]
pub struct Emulator {{
    cycles: u64,
    rom: Vec<u8>,
    framebuffer: Vec<u8>,
    keys: [bool; 16],
}}

impl Default for Emulator {{
    fn default() -> Self {{
        Self::new()
    }}
}}

impl Emulator {{
    pub const WIDTH: usize = {width};
    pub const HEIGHT: usize = {height};
    pub const SMOKE_CYCLES: u64 = {cycles};

    pub fn new() -> Self {{
        Self {{
            cycles: 0,
            rom: Vec::new(),
            framebuffer: vec![0; Self::WIDTH * Self::HEIGHT],
            keys: [false; 16],
        }}
    }}

    pub fn reset(&mut self) {{
        self.cycles = 0;
        self.framebuffer.fill(0);
        self.keys = [false; 16];
    }}

    pub fn load_rom(&mut self, rom: &[u8]) -> Result<(), EmulatorError> {{
        self.rom.clear();
        self.rom.extend_from_slice(rom);
        self.reset();
        Ok(())
    }}

    pub fn step(&mut self) -> Result<(), EmulatorError> {{
        self.cycles = self.cycles.wrapping_add(1);
        Ok(())
    }}

    pub fn run_cycles(&mut self, cycles: u64) -> Result<(), EmulatorError> {{
        for _ in 0..cycles {{
            self.step()?;
        }}
        Ok(())
    }}

    pub fn framebuffer(&self) -> &[u8] {{
        &self.framebuffer
    }}

    pub fn set_key(&mut self, key: usize, pressed: bool) {{
        if key < self.keys.len() {{
            self.keys[key] = pressed;
        }}
    }}

    pub fn save_state(&self) -> Snapshot {{
        Snapshot {{
            cycles: self.cycles,
            framebuffer: self.framebuffer.clone(),
        }}
    }}

    pub fn load_state(&mut self, snapshot: &Snapshot) -> Result<(), EmulatorError> {{
        self.cycles = snapshot.cycles;
        self.framebuffer = snapshot.framebuffer.clone();
        Ok(())
    }}

    pub fn cycles(&self) -> u64 {{
        self.cycles
    }}
}}
"""


def starter_main_file(task: ConfigData, state: ConfigData | None = None) -> str:
    _ = state
    info = task["info"]
    platform = str(info["slug"])
    smoke_cycles = int(info.get("runtime", {}).get("smoke_cycles", 1024))
    return f"""\
use emulator_benchmark::Emulator;
use std::collections::hash_map::DefaultHasher;
use std::hash::{{Hash, Hasher}};

fn hash_bytes(bytes: &[u8]) -> String {{
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    format!("{{:016x}}", hasher.finish())
}}

fn json_arg(args: &[String], name: &str) -> Option<String> {{
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].clone())
}}

fn main() {{
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--contract-version") {{
        println!("emulator-runner-v1");
        return;
    }}
    if args.iter().any(|arg| arg == "--self-test") {{
        println!("{{{{\\\"ok\\\":true,\\\"contract\\\":\\\"emulator-runner-v1\\\"}}}}");
        return;
    }}
    if let Some(output_path) = json_arg(&args, "--output") {{
        let case_id = json_arg(&args, "--case").unwrap_or_else(|| "unknown".to_string());
        let cycles: u64 = json_arg(&args, "--cycles")
            .and_then(|value| value.parse().ok())
            .unwrap_or({smoke_cycles});
        let mut emulator = Emulator::new();
        if let Some(rom_path) = json_arg(&args, "--rom") {{
            let rom = std::fs::read(rom_path).unwrap_or_default();
            emulator.load_rom(&rom).expect("load_rom");
        }}
        emulator.run_cycles(cycles).expect("run_cycles");
        let frame_hash = hash_bytes(emulator.framebuffer());
        let payload = format!(
            "{{{{\\\"platform\\\":\\\"{platform}\\\",\\\"case_id\\\":\\\"{{}}\\\",\\\"passed\\\":false,\\\"cycles\\\":{{}},\\\"frames\\\":1,\\\"framebuffer_sha256\\\":\\\"{{}}\\\",\\\"serial\\\":\\\"starter scaffold does not implement public ROM suite\\\"}}}}",
            case_id,
            emulator.cycles(),
            frame_hash
        );
        std::fs::write(output_path, payload).expect("write output");
        return;
    }}

    let mut emulator = Emulator::new();
    let rom: Vec<u8> = (0u8..=31).collect();
    emulator.load_rom(&rom).expect("load_rom");
    emulator.run_cycles({smoke_cycles}).expect("run_cycles");
    let frame_hash = hash_bytes(emulator.framebuffer());
    println!(
        "{{{{\\\"platform\\\":\\\"{platform}\\\",\\\"cycles\\\":{{}},\\\"framebuffer_hash\\\":\\\"{{}}\\\",\\\"deterministic\\\":true}}}}",
        emulator.cycles(),
        frame_hash
    );
}}
"""


def public_contract_file(task: ConfigData, state: ConfigData | None = None) -> str:
    _ = state
    info = task["info"]
    width = int(info.get("framebuffer", {}).get("width", 160))
    height = int(info.get("framebuffer", {}).get("height", 144))
    return f"""\
use emulator_benchmark::Emulator;

#[test]
fn exposes_required_core_api() {{
    let mut emulator = Emulator::new();
    emulator.reset();
    emulator.load_rom(&[0x00, 0x01, 0x02, 0x03]).unwrap();
    emulator.step().unwrap();
    emulator.run_cycles(8).unwrap();
    emulator.set_key(0, true);
    let snapshot = emulator.save_state();
    emulator.load_state(&snapshot).unwrap();
    assert_eq!(emulator.framebuffer().len(), {width * height});
}}

#[test]
fn deterministic_replay_from_same_rom() {{
    let rom = [0x42, 0x99, 0x10, 0x77, 0x00, 0xFE];
    let mut first = Emulator::new();
    let mut second = Emulator::new();
    first.load_rom(&rom).unwrap();
    second.load_rom(&rom).unwrap();
    first.run_cycles(32).unwrap();
    second.run_cycles(32).unwrap();
    assert_eq!(first.framebuffer(), second.framebuffer());
    assert_eq!(first.cycles(), second.cycles());
}}
"""


def workspace_readme_file(task: ConfigData, state: ConfigData | None = None) -> str:
    _ = state
    info = task["info"]
    public = "\n".join(f"- {item}" for item in info["public_verification"])
    hidden = "\n".join(f"- {item}" for item in info["hidden_verification"])
    return f"""\
# {info["display_name"]} Emulator Task

Implement the emulator in this Rust crate. The public contract tests in
`tests/public_contract.rs` check the required harness API. The benchmark grader
also checks deterministic command output, required platform terminology, and
the verification targets below.

{runner_contract_markdown()}

## Public verification targets

{public}

## Hidden verification categories

{hidden}
"""


def public_manifest_file(task: ConfigData, state: ConfigData | None = None) -> str:
    _ = state
    payload = {
        "platform": task["info"]["slug"],
        "public_cases": task["info"]["public_cases"],
        "framebuffer": task["info"].get("framebuffer", {}),
        "runtime": task["info"].get("runtime", {}),
        "runner_contract": runner_contract(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def hidden_manifest_json(task: ConfigData) -> str:
    payload = {
        "platform": task["info"]["slug"],
        "display_name": task["info"]["display_name"],
        "requirements": task["info"]["requirements"],
        "public_cases": task["info"]["public_cases"],
        "hidden_cases": task["info"]["hidden_cases"],
        "framebuffer": task["info"].get("framebuffer", {}),
        "runtime": task["info"].get("runtime", {}),
        "scoring_weights": task["info"].get("scoring_weights", {}),
        "runner_contract": runner_contract(),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_program_config(manifest: EmulatorManifest) -> ProgramData:
    _ = manifest
    return {
        "files": {
            "/workspace/Cargo.toml": cargo_toml_file,
            "/workspace/src/lib.rs": starter_lib_file,
            "/workspace/src/main.rs": starter_main_file,
            "/workspace/tests/public_contract.rs": public_contract_file,
            "/workspace/verification/public_manifest.json": public_manifest_file,
            "/workspace/verification/runner_contract.json": lambda task, state=None: (
                json.dumps(runner_contract(), indent=2, sort_keys=True)
            ),
            "/workspace/README.md": workspace_readme_file,
        },
        "setup": [
            "mkdir -p /workspace/src /workspace/tests /workspace/verification",
            (
                "printf '%s\\n' "
                "'export PATH=/usr/local/cargo/bin:/root/.cargo/bin:/usr/local/bin:$PATH' "
                "'export CARGO_HOME=/usr/local/cargo' "
                "'export PAGER=cat' "
                "'export MANPAGER=cat' "
                "> /etc/profile.d/emulator_toolchain.sh || true"
            ),
        ],
        "env": {
            "AGENT_WORKDIR": "/workspace",
            "PATH": (
                "/usr/local/cargo/bin:/root/.cargo/bin:/usr/local/sbin:"
                "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "CARGO_HOME": "/usr/local/cargo",
            "PAGER": "cat",
            "MANPAGER": "cat",
        },
    }


VERIFIER_SCRIPT = textwrap.dedent(
    r"""
    import argparse
    import json
    import subprocess
    import sys
    from pathlib import Path


    def main() -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--workspace", required=True)
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--sources", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()

        sys.path.insert(0, "/tmp")
        from emulator_suite_adapters import verify_workspace

        workspace = Path(args.workspace)
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        with open(args.sources, encoding="utf-8") as f:
            sources = json.load(f)
        result = verify_workspace(workspace, manifest, sources)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"score": result["score"], "passed": result["passed"]}, sort_keys=True))
        return 0 if result["score"] > 0 else 1


    if __name__ == "__main__":
        sys.exit(main())
    """
).strip()
