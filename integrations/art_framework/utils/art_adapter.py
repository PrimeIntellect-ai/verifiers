from typing import Any, Callable, Tuple

from datasets import Dataset


def _compile_lambda(src: str) -> Callable:
    # Very small, controlled lambda support for examples; fail fast otherwise
    src = src.strip()
    if not (src.startswith("lambda ") and (":" in src)):
        raise ValueError("Only simple lambda implementations are supported in examples")
    return eval(src, {"__builtins__": {}})


def art_tool_to_callable(tool_spec: dict) -> Callable:
    name = tool_spec.get("name")
    description = tool_spec.get("description", "")
    parameters = tool_spec.get("parameters", {"type": "object", "properties": {}})
    implementation = tool_spec.get("implementation")

    if implementation is None:
        raise ValueError(f"Tool {name} missing implementation")

    impl = implementation
    if isinstance(implementation, str):
        impl = _compile_lambda(implementation)

    # Build a strongly-typed function with explicit parameters (no **kwargs)
    props: dict = (
        parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    )
    required: list[str] = (
        parameters.get("required", []) if isinstance(parameters, dict) else []
    )

    def _map_type(t: str) -> str:
        return {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }.get(t, "Any")

    # Order params: required first, then optional (default=None)
    ordered_keys = [k for k in required if k in props] + [
        k for k in props.keys() if k not in required
    ]
    annot_params: list[str] = []
    call_kwargs: list[str] = []
    for key in ordered_keys:
        spec = props.get(key, {}) or {}
        tname = _map_type(str(spec.get("type", "Any")))
        if key in required:
            annot_params.append(f"{key}: {tname}")
        else:
            annot_params.append(f"{key}: {tname} | None = None")
        call_kwargs.append(f"{key}={key}")

    params_sig = ", ".join(annot_params)
    call_sig = ", ".join(call_kwargs)
    fn_src = f"def {name}({params_sig}):\n    return __impl({call_sig})\n"
    ns: dict[str, Any] = {"__impl": impl, "Any": Any}
    exec(fn_src, ns)
    typed_fn = ns[name]
    typed_fn.__name__ = name
    typed_fn.__doc__ = description
    setattr(
        typed_fn,
        "__art_schema__",
        {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    )
    return typed_fn


def art_config_to_tools(config: dict) -> list[Callable]:
    tools = config.get("tools", [])
    return [art_tool_to_callable(t) for t in tools]


def get_completion_tool_name(config: dict) -> str:
    name = config.get("completion_tool_name")
    if not isinstance(name, str) or not name:
        raise ValueError("completion_tool_name must be a non-empty string")
    return name


def build_dataset_from_art_config(config: dict) -> Tuple[Dataset, Dataset]:
    # Minimal examples: if provided under config["examples"], split 2/2 else create trivial
    examples = config.get("examples") or []
    if not examples:
        # trivial calculator example
        examples = [
            {"question": "Add 2 and 3", "answer": "5"},
            {"question": "Add 10 and 1", "answer": "11"},
            {"question": "Add 4 and 4", "answer": "8"},
            {"question": "Add 7 and 3", "answer": "10"},
        ]
    # ensure fields
    rows = []
    sys_prompt = config.get("system_prompt") or "Use tools to solve the task."
    for ex in examples:
        question = ex.get("question") or ex.get("prompt") or ""
        answer = str(ex.get("answer", ""))
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ],
                "answer": answer,
            }
        )
    # split 2 train / 2 eval by default
    split = max(1, min(2, len(rows) // 2))
    train_rows = rows[:split]
    eval_rows = rows[split : split * 2] or rows[:split]
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)
