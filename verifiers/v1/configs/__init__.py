"""The config tree, separated from the logic it configures: one module per
domain (`configs.agent`, `configs.harness`, ...) plus the CLI entrypoints' run
configs under `configs.cli`. Config modules import only types and other configs,
so any module — the trace record included — can embed them without cycles."""
