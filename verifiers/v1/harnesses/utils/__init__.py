"""Shared building blocks for harness implementations.

`mcp`, `compaction`, and `core` are written to run inside bundled PEP 723
programs: `launch.bundle_program` splices their sources into a program script,
so they import only the packages a program declares and reference each other
through `TYPE_CHECKING` imports that the flat bundle resolves at runtime.
"""
