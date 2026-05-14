# v1 Environment Best Practices

This is the working checklist for building environments with the v1
Taskset/Harness pattern.

## Do

- Expose `load_environment(config: vf.EnvConfig) -> vf.Env` for v1
  environments. The loader receives a typed config object from the caller.
- Import the public API with `import verifiers as vf`. `verifiers.v1` remains
  available for framework-internal tests and narrow module-level checks, but
  user environment code should use the top-level namespace.
- Use typed Pydantic config objects in Python code. Raw mappings are for TOML,
  CLI, and other external boundaries.
- Keep `config` parameters to one concrete Pydantic config type or `None`.
  Do not advertise unions of mappings, base configs, and specific configs.
- Treat `Mapping[str, object]` as an explicit boundary type. Accept it only for
  intentionally dynamic payloads such as task rows, protocol messages,
  sandbox/program specs, or Pydantic config fields that store arbitrary user
  objects. Prefer a named alias such as `ConfigMap`, `TaskRow`, or
  `Objects` over spelling the broad type in user-facing
  signatures.
- Do not use raw `Any` in v1 environment code. If a value is intentionally
  arbitrary, give that boundary a named type in `verifiers.v1.types`.
- Put task/data/scoring settings on a `TasksetConfig` owned by the taskset.
- Put rollout program/runtime settings on a `HarnessConfig` owned by the
  harness. For example, `vf.RLM` takes `vf.RLMConfig`.
- Keep environment files as wiring: taskset construction, harness construction,
  and small policy choices that compose the two.
- Prefer v1-native framework objects over stdlib-shaped user code. User-facing
  environment code should read as Verifiers code first.
- Use `Taskset(objects=..., bindings=...)` for shared extractors, judges,
  clients, and other dependencies that signal functions need.
- Compose related categories inside one taskset only when they share the same
  harness lifecycle and scoring contract.
- Expose explicit typed loaders for separate v1 envs when categories need
  different tasksets, harnesses, or lifecycle behavior.
- Keep dual v0/v1 loaders explicit only while migration is intentionally
  dual-stack.

## Don't

- Do not mirror every taskset or harness config field as `load_environment`
  kwargs.
- Do not put harness settings on taskset configs or taskset settings on harness
  configs.
- Do not add environment-level config subclasses to carry fields already owned
  by a taskset or harness config.
- Do not wrap v1 `Env` objects in the v0 `EnvGroup`.
- Do not add thin intermediate harness types that only restate `Harness`.
  Reusable command agents should be direct `Harness` subclasses with their own
  typed config only when they have real behavior to own.
- Do not add heterogeneous `TasksetGroup` routing as a substitute for a real
  v1 suite abstraction.
- Do not overfit v1 APIs to names or layering inherited from
  `research-environments`; keep the logical v1 ownership boundaries correct.
- Do not use vague config names that only repeat the component name. Name the
  actual thing being configured: `rlm_repo_ref`, not `rlm_ref`; `env_vars`, not
  `rlm_env`.
- Do not write global helper functions in environment files. Rare exceptions
  are process-level handles, such as a lock, where a module-level object is the
  cleanest way to assert process-wide control.
- Do not make users manipulate paths, package resources, or other stdlib details
  when the framework can express the intent directly.

## Enforcement

- Put deterministic, repo-specific style rules in Semgrep rather than custom
  AST scripts. Ruff owns generic lint and format, ty owns type correctness, and
  Semgrep owns Verifiers-specific policy checks.
- Keep Semgrep rules principle-based. They should encode stable contracts such
  as narrow config parameters and no broad user-facing `Any`, not one-off bans
  for removed historical names.
