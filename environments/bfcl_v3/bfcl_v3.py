import json
import re
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import Field, model_validator

import verifiers as vf
from verifiers.types import (
    AssistantMessage,
    MessageContent,
    Messages,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.message_utils import message_role, normalize_messages

from verifiers.v1.utils.endpoint_utils import assistant_completion_from_messages
from verifiers.v1.utils.json_utils import json_args
from verifiers.v1.types import ConfigMap

BFCL_TOOLSET_REF = "bfcl_v3:load_bfcl_toolset"
_BFCL_PATCHED = False
BFCLRawMessage = str | ConfigMap
BFCLRawTurn = str | ConfigMap | Sequence[BFCLRawMessage] | None


class BFCLTasksetConfig(vf.TasksetConfig):
    test_category: str = "simple_python"
    examples_per_category: int = -1


class BFCLHarnessConfig(vf.HarnessConfig):
    test_category: str = "simple_python"


class BFCLEnvConfig(vf.EnvConfig):
    taskset: BFCLTasksetConfig = Field(default_factory=BFCLTasksetConfig)
    harness: BFCLHarnessConfig = Field(default_factory=BFCLHarnessConfig)

    @model_validator(mode="after")
    def sync_harness_category(self) -> "BFCLEnvConfig":
        self.harness = BFCLHarnessConfig(
            self.harness, test_category=self.taskset.test_category
        )
        return self


def modded_convert_func_name(function_name: str, model_name: str) -> str:
    _ = model_name
    return re.sub(r"\.", "_", function_name)


def patch_bfcl_eval() -> None:
    global _BFCL_PATCHED
    if _BFCL_PATCHED:
        return
    import bfcl_eval.constants.category_mapping as category_mapping
    import bfcl_eval.eval_checker.ast_eval.ast_checker as ast_checker_module

    agentic_categories = category_mapping.AGENTIC_CATEGORY.copy()
    category_mapping.AGENTIC_CATEGORY.clear()
    for category in agentic_categories:
        if category in category_mapping.ALL_SCORING_CATEGORIES:
            category_mapping.ALL_SCORING_CATEGORIES.remove(category)
        if category in category_mapping.ALL_CATEGORIES:
            category_mapping.ALL_CATEGORIES.remove(category)
    category_mapping.TEST_COLLECTION_MAPPING.pop("memory", None)
    category_mapping.TEST_COLLECTION_MAPPING.pop("web_search", None)
    category_mapping.TEST_COLLECTION_MAPPING.pop("agentic", None)

    non_scoring_categories = category_mapping.NON_SCORING_CATEGORY.copy()
    category_mapping.NON_SCORING_CATEGORY.clear()
    for category in non_scoring_categories:
        if category in category_mapping.ALL_CATEGORIES:
            category_mapping.ALL_CATEGORIES.remove(category)
    category_mapping.TEST_COLLECTION_MAPPING.pop("format_sensitivity", None)

    setattr(ast_checker_module, "convert_func_name", modded_convert_func_name)
    _BFCL_PATCHED = True


def bfcl_tool_defs(functions: object) -> list[Tool]:
    patch_bfcl_eval()
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.model_handler.utils import convert_to_tool

    oai_tools = convert_to_tool(
        functions, GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS
    )
    tool_defs = []
    for tool in oai_tools:
        function = tool["function"]
        tool_defs.append(
            Tool(
                name=str(function["name"]),
                description=str(function.get("description") or ""),
                parameters=dict(cast(ConfigMap, function["parameters"])),
                strict=False,
            )
        )
    return tool_defs


class BFCLSchemaTool:
    def __init__(self, tool_def: Tool):
        self.name = tool_def.name
        self.__name__ = tool_def.name
        self.__doc__ = tool_def.description
        self.tool_def = tool_def

    async def __call__(self, state: vf.State, **arguments: object) -> str:
        calls = cast(
            list[vf.ConfigData], state.setdefault("bfcl_executed_tool_calls", [])
        )
        calls.append({self.name: arguments})
        return "recorded"


def load_bfcl_toolset(task: ConfigMap) -> vf.Toolset:
    return vf.Toolset(
        tools=[
            BFCLSchemaTool(tool_def)
            for tool_def in bfcl_tool_defs(bfcl_functions(task))
        ]
    )


def bfcl_functions(task: ConfigMap) -> object:
    return task.get("function_with_hints") or task["function"]


def bfcl_missed_function(task: ConfigMap) -> ConfigMap:
    value = task.get("missed_function_with_hints") or task.get("missed_function") or {}
    if not isinstance(value, Mapping):
        raise TypeError("BFCL missed_function must be a mapping.")
    return cast(ConfigMap, value)


def build_source(test_category: str, examples_per_category: int = -1):
    def source():
        patch_bfcl_eval()
        from bfcl_eval.utils import (
            is_multi_turn,
            is_relevance_or_irrelevance,
            load_dataset_entry,
            load_ground_truth_entry,
        )

        entries = load_dataset_entry(
            test_category, include_language_specific_hint=False
        )
        entries_with_hints = load_dataset_entry(
            test_category, include_language_specific_hint=True
        )
        if is_relevance_or_irrelevance(test_category):
            ground_truth_entries = [None] * len(entries)
        else:
            ground_truth_entries = load_ground_truth_entry(test_category)
        limit = len(entries) if examples_per_category < 0 else examples_per_category
        rows = []
        for index, (entry, hinted_entry, ground_truth) in enumerate(
            zip(entries, entries_with_hints, ground_truth_entries)
        ):
            if index >= limit:
                break
            row = bfcl_row(
                test_category,
                entry,
                hinted_entry,
                cast(ConfigMap | None, ground_truth),
            )
            if is_multi_turn(test_category):
                max_steps = maximum_step_limit()
                row["max_steps_per_turn"] = max_steps
                row["max_turns"] = (
                    len(cast(Sequence[BFCLRawTurn], row["question"])) * max_steps
                )
            else:
                row["max_turns"] = 1
                row["toolsets"] = {"bfcl": {"fn": BFCL_TOOLSET_REF}}
            rows.append(row)
        return rows

    return source


def bfcl_row(
    test_category: str,
    entry: ConfigMap,
    hinted_entry: ConfigMap,
    ground_truth: ConfigMap | None,
) -> vf.ConfigData:
    question = cast(list[BFCLRawTurn], entry["question"])
    first_turn_system_prompt, first_turn_prompt = split_system_prompt(
        normalize_turn(question[0])
    )
    row: vf.ConfigData = {
        "task_id": str(entry["id"]),
        "id": str(entry["id"]),
        "category": test_category,
        "prompt": first_turn_prompt,
        "question": [
            first_turn_prompt,
            *[normalize_turn(turn) for turn in question[1:]],
        ],
        "function": entry["function"],
        "function_with_hints": hinted_entry["function"],
    }
    if first_turn_system_prompt:
        row["system_prompt"] = first_turn_system_prompt
    for key in (
        "initial_config",
        "involved_classes",
    ):
        if key in entry:
            row[key] = entry[key]
    if "missed_function" in entry:
        row["missed_function"] = entry["missed_function"]
    if "missed_function" in hinted_entry:
        row["missed_function_with_hints"] = hinted_entry["missed_function"]
    if ground_truth is not None:
        for key, value in ground_truth.items():
            row[key] = value
    return row


def normalize_turn(value: object) -> list[vf.ConfigData]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, Mapping):
        return [dict(cast(ConfigMap, value))]
    if isinstance(value, Sequence):
        messages = []
        for item in value:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, Mapping):
                messages.append(dict(cast(ConfigMap, item)))
            else:
                raise TypeError(f"Unsupported BFCL message item: {type(item).__name__}")
        return messages
    raise TypeError(f"Unsupported BFCL prompt turn: {type(value).__name__}")


def split_system_prompt(
    messages: Sequence[ConfigMap],
) -> tuple[list[vf.ConfigData], list[vf.ConfigData]]:
    system_prompt = []
    prompt = []
    for message in messages:
        target = system_prompt if message.get("role") == "system" else prompt
        target.append(dict(message))
    return system_prompt, prompt


def maximum_step_limit() -> int:
    patch_bfcl_eval()
    from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT

    return cast(int, MAXIMUM_STEP_LIMIT)


def model_name(state: ConfigMap) -> str:
    runtime = state.get("runtime") or {}
    if isinstance(runtime, Mapping):
        runtime_map = cast(ConfigMap, runtime)
        model = runtime_map.get("model")
        if isinstance(model, str) and model:
            return model
    return "unknown"


def assistant_tool_calls(state: ConfigMap) -> list[ToolCall]:
    completion = state.get("completion") or []
    if not isinstance(completion, Sequence):
        return []
    messages = vf.get_messages(completion, role="assistant")
    if not messages:
        return []
    return parse_tool_calls(messages[-1])


def parse_tool_calls(message: object) -> list[ToolCall]:
    if isinstance(message, AssistantMessage):
        return list(message.tool_calls or [])
    raw_tool_calls: object
    if isinstance(message, Mapping):
        message_map = cast(ConfigMap, message)
        raw_tool_calls = message_map.get("tool_calls") or []
    else:
        raw_tool_calls = getattr(message, "tool_calls", []) or []
    if not isinstance(raw_tool_calls, Sequence):
        return []
    calls = []
    for raw_call in raw_tool_calls:
        if isinstance(raw_call, ToolCall):
            calls.append(raw_call)
            continue
        if not isinstance(raw_call, Mapping):
            continue
        raw_call = cast(ConfigMap, raw_call)
        function = raw_call.get("function")
        if isinstance(function, Mapping):
            function_map = cast(ConfigMap, function)
            name = str(function_map.get("name") or "")
            arguments = function_map.get("arguments") or "{}"
        else:
            name = str(raw_call.get("name") or "")
            arguments = raw_call.get("arguments") or "{}"
        if not name:
            continue
        calls.append(
            ToolCall(
                id=str(raw_call.get("id") or name),
                name=name,
                arguments=arguments
                if isinstance(arguments, str)
                else json.dumps(arguments),
            )
        )
    return calls


def convert_to_gorilla(tool_calls: list[ToolCall]) -> list[vf.ConfigData]:
    decoded_output = []
    for tool_call in tool_calls:
        decoded_output.append({tool_call.name: json_args(tool_call.arguments)})
    return decoded_output


def convert_to_func_calls(tool_calls: list[ToolCall]) -> list[str]:
    func_calls = []
    for tool_call in tool_calls:
        params = json_args(tool_call.arguments)
        args = ",".join(f"{key}={value!r}" for key, value in params.items())
        func_calls.append(f"{tool_call.name}({args})")
    return func_calls


def json_clone(value: object) -> object:
    return json.loads(json.dumps(value))


@vf.reward(weight=1.0)
async def bfcl_reward(task: ConfigMap, state: vf.State) -> float:
    patch_bfcl_eval()
    from bfcl_eval.utils import is_multi_turn, is_relevance_or_irrelevance

    category = str(task["category"])
    if is_relevance_or_irrelevance(category):
        return relevance_reward(task, state)
    if is_multi_turn(category):
        return multi_turn_reward(task, state)
    return ast_reward(task, state)


def relevance_reward(task: ConfigMap, state: ConfigMap) -> float:
    patch_bfcl_eval()
    from bfcl_eval.utils import is_empty_output

    category = str(task["category"])
    try:
        gorilla_tool_calls = convert_to_gorilla(assistant_tool_calls(state))
        contain_func_call = not is_empty_output(gorilla_tool_calls)
    except Exception:
        contain_func_call = False
    if "irrelevance" in category:
        return float(not contain_func_call)
    return float(contain_func_call)


def ast_reward(task: ConfigMap, state: ConfigMap) -> float:
    patch_bfcl_eval()
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.utils import (
        is_function_calling_format_output,
        is_java,
        is_js,
    )

    category = str(task["category"])
    try:
        gorilla_tool_calls = convert_to_gorilla(assistant_tool_calls(state))
        if not is_function_calling_format_output(gorilla_tool_calls):
            return 0.0
    except Exception:
        return 0.0

    if is_java(category):
        language = Language.JAVA
    elif is_js(category):
        language = Language.JAVASCRIPT
    else:
        language = Language.PYTHON

    checker_result = ast_checker(
        task["function"],
        gorilla_tool_calls,
        task["ground_truth"],
        language,
        category,
        model_name(state),
    )
    return float(bool(checker_result["valid"]))


def multi_turn_reward(task: ConfigMap, state: ConfigMap) -> float:
    patch_bfcl_eval()
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )
    from bfcl_eval.model_handler.base_handler import is_empty_execute_response

    completion = state.get("completion") or []
    if not isinstance(completion, Sequence):
        return 0.0
    raw_ground_truth = task["ground_truth"]
    if not isinstance(raw_ground_truth, Sequence):
        return 0.0
    all_ground_truth = cast(list[list[str]], raw_ground_truth)
    all_func_calls: list[list[list[str]]] = [[]]
    try:
        for message in completion:
            role = message_role(message)
            if role == "user":
                all_func_calls.append([])
            elif role == "tool":
                continue
            elif role == "assistant":
                func_calls = convert_to_func_calls(parse_tool_calls(message))
                if is_empty_execute_response(func_calls):
                    continue
                all_func_calls[-1].append(func_calls)
            elif role == "system":
                continue
            else:
                return 0.0
    except Exception:
        return 0.0

    if len(all_func_calls) != len(all_ground_truth):
        return 0.0

    result = multi_turn_checker(
        all_func_calls,
        all_ground_truth,
        {
            "initial_config": task.get("initial_config", {}),
            "involved_classes": task["involved_classes"],
            "id": task["id"],
        },
        str(task["id"]).rsplit("_", 1)[0],
        model_name(state),
    )
    return float(bool(result["valid"]))


async def bfcl_multi_turn_program(
    task: vf.Task, state: vf.State, harness: vf.Harness
) -> vf.State:
    patch_bfcl_eval()
    from bfcl_eval.constants.default_prompts import (
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
    from bfcl_eval.model_handler.base_handler import is_empty_execute_response

    messages = [
        *normalize_messages(
            state.get("system_prompt", []), field_name="state.system_prompt"
        ),
        *normalize_messages(state.get("prompt", []), field_name="state.prompt"),
    ]
    prompt_messages = [message.model_dump(exclude_none=True) for message in messages]

    def sync_completion() -> list[vf.ConfigData]:
        rendered_messages = [
            message.model_dump(exclude_none=True) for message in messages
        ]
        state["completion"] = assistant_completion_from_messages(
            prompt_messages, rendered_messages
        )
        return rendered_messages

    category = str(task["category"])
    tool_defs = bfcl_tool_defs(bfcl_functions(task))
    next_prompts = list(cast(Sequence[list[vf.ConfigData]], task["question"]))[1:]
    holdout_function = bfcl_missed_function(task)
    initial_config = cast(vf.ConfigData, json_clone(task.get("initial_config") or {}))
    involved_classes = cast(list[str], json_clone(task["involved_classes"]))
    max_steps_per_turn = int(task.get("max_steps_per_turn") or maximum_step_limit())
    turn_idx = 0
    steps_per_turn = 0
    runtime = harness.runtime
    max_model_requests = state.get_max_turns(harness.config.max_turns)
    model_requests = 0

    execute_multi_turn_func_call(
        [],
        initial_config,
        involved_classes,
        model_name(state).replace("/", "_").replace("-", "_").replace(".", "_"),
        str(task["id"]),
        long_context=("long_context" in category or "composite" in category),
    )

    while max_model_requests <= 0 or model_requests < max_model_requests:
        if await runtime.is_completed(task, state):
            return state
        response = await runtime.submit_model_request(
            cast(Messages, messages),
            task,
            state,
            tool_defs=tool_defs,
        )
        model_requests += 1
        messages.append(response.message)
        sync_completion()
        tool_calls = list(response.message.tool_calls or [])
        try:
            func_calls = convert_to_func_calls(tool_calls)
            if is_empty_execute_response(func_calls):
                func_calls = None
        except Exception:
            func_calls = None

        if func_calls:
            execution_results, _ = execute_multi_turn_func_call(
                func_call_list=func_calls,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=model_name(state)
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_"),
                test_entry_id=str(task["id"]),
                long_context=("long_context" in category or "composite" in category),
            )
            for execution_result, tool_call in zip(execution_results, tool_calls):
                messages.append(
                    ToolMessage(
                        tool_call_id=tool_call.id,
                        content=cast(MessageContent, execution_result),
                    )
                )
                sync_completion()
            steps_per_turn += 1
            if steps_per_turn >= max_steps_per_turn:
                state.stop("max_steps_per_turn_reached")
                return state
            continue

        steps_per_turn = 0
        turn_idx += 1
        if not next_prompts:
            state.stop("no_next_prompt_and_no_tool_calls")
            return state
        next_prompt = normalize_turn(next_prompts.pop(0))
        if str(turn_idx) in holdout_function:
            tool_defs.extend(bfcl_tool_defs(holdout_function[str(turn_idx)]))
            if next_prompt:
                raise ValueError("BFCL holdout turns must not include user messages.")
            messages.append(
                UserMessage(content=DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC)
            )
        else:
            messages.extend(normalize_messages(cast(Messages, next_prompt)))
        sync_completion()

    state.stop("max_turns_reached")
    return state


class BFCLHarness(vf.Harness):
    def __init__(self, config: BFCLHarnessConfig | None = None):
        config = BFCLHarnessConfig(config)
        patch_bfcl_eval()
        from bfcl_eval.utils import is_multi_turn

        program = (
            self.run_bfcl_multi_turn if is_multi_turn(config.test_category) else None
        )
        super().__init__(program=program, config=config)

    async def run_bfcl_multi_turn(self, task: vf.Task, state: vf.State) -> vf.State:
        return await bfcl_multi_turn_program(task, state, self)


class BFCLTaskset(vf.Taskset):
    def __init__(self, config: BFCLTasksetConfig | None = None):
        config = BFCLTasksetConfig(config)
        super().__init__(
            source=build_source(config.test_category, config.examples_per_category),
            rewards=[bfcl_reward],
            config=config,
        )


def load_taskset(config: BFCLTasksetConfig) -> BFCLTaskset:
    return BFCLTaskset(config=config)


def load_harness(config: BFCLHarnessConfig) -> BFCLHarness:
    return BFCLHarness(config=config)


def load_environment(
    config: BFCLEnvConfig,
) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
