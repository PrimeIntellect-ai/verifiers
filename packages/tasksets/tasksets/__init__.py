"""Shipped v1 tasksets, resolved by id (`--taskset.id <id>`).

`REGISTRY` maps a taskset id to its dotted module (each exposes `load_taskset(config)`).
The loader imports the module on demand, so importing this package stays cheap — e.g. the
textarena taskset's `textarena` dependency is only imported when that id is selected.
"""

REGISTRY: dict[str, str] = {
    "harbor": "tasksets.harbor",
    "textarena_v1": "tasksets.textarena_v1",
}
