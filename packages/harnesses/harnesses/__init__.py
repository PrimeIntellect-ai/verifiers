"""Shipped v1 harnesses, resolved by id (`--harness.id <id>`).

`REGISTRY` maps a harness id to its dotted module (each exposes `load_harness(config)`).
The loader imports the module on demand, so importing this package stays cheap.
"""

REGISTRY: dict[str, str] = {
    "default": "harnesses.default",
    "rlm": "harnesses.rlm",
}
