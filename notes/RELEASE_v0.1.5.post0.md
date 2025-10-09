# Verifiers v0.1.5 Release Notes

*Date:* 10/8/25

***Update***: 10/9/25
- Update SandboxEnv to use prime-sandboxes instead of prime-cli, disable Python 3.14 as `datasets` is not compatible (blocked by `arrow`).

Verifiers v0.1.5 adds a new SandboxEnv environment for running code in a sandboxed environment, along with a number of bug fixes and QoL improvements.

## Highlights

- New SandboxEnv environment for running code in a sandboxed environment. This environment is configured for use with Prime Intellect's sandboxes, but can be adapted for other sandbox providers.
- Improvements to `StatefulToolEnv` and `ToolEnv` for handling stateful tools and tools with arguments that should not be passed to the model, and managing active tools + schemas (`add_tool` and `remove_tool`).
- Bug fixes and improvements (`vf-tui`, tool call serialization, documentation, examples).


**Full Changelog**: https://github.com/willccbb/verifiers/compare/v0.1.4...v0.1.5