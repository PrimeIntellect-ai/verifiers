import json
import re
import time
from collections.abc import Sequence
from typing import cast

from pydantic import Field, model_validator

import verifiers.v1 as vf
from verifiers.types import (
    AssistantMessage,
    Message,
    MessageContent,
    Messages,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from verifiers.utils.response_utils import parse_response_message

_BFCL_PATCHED = False
BFCLRawMessage = str | vf.JsonData
BFCLRawTurn = str | vf.JsonData | Sequence[BFCLRawMessage] | None


class BFCLTask(vf.Task):
    category: str
    question: list[list[vf.JsonData]]
    function: vf.JsonValue
    function_with_hints: vf.JsonValue | None = None
    ground_truth: vf.JsonValue | None = None
    initial_config: vf.JsonData = Field(default_factory=dict)
    involved_classes: vf.JsonValue | None = None
    missed_function: vf.JsonData = Field(default_factory=dict)
    missed_function_with_hints: vf.JsonData = Field(default_factory=dict)
    max_steps_per_turn: int | None = None


class BFCLTasksetConfig(vf.TasksetConfig):
    id: str = "bfcl-v3"
    test_category: str = "simple_python"
    test_categories: list[str] | None = None
    examples_per_category: int = -1

    @model_validator(mode="after")
    def validate_category_routing(self) -> "BFCLTasksetConfig":
        if self.test_categories is not None:
            raise ValueError(
                "BFCL v1 accepts one test_category per taskset. Configure separate "
                "evals for multiple categories."
            )
        return self


class BFCLHarnessConfig(vf.HarnessConfig):
    max_turns: int = 1


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
    tool_defs: list[Tool] = []
    for tool in oai_tools:
        function = tool["function"]
        tool_defs.append(
            Tool(
                name=str(function["name"]),
                description=str(function.get("description") or ""),
                parameters=dict(cast(vf.JsonData, function["parameters"])),
                strict=False,
            )
        )
    return tool_defs


def bfcl_functions(task: vf.Task) -> object:
    task = cast(BFCLTask, task)
    return (
        task.function_with_hints
        if task.function_with_hints is not None
        else task.function
    )


def bfcl_missed_function(task: vf.Task) -> vf.JsonData:
    task = cast(BFCLTask, task)
    return task.missed_function_with_hints or task.missed_function


def build_task_loader(test_category: str, examples_per_category: int = -1):
    def factory() -> list[vf.JsonData]:
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
        rows: list[vf.JsonData] = []
        for index, (entry, hinted_entry, ground_truth) in enumerate(
            zip(entries, entries_with_hints, ground_truth_entries)
        ):
            if index >= limit:
                break
            row = bfcl_row(
                test_category,
                cast(vf.JsonData, entry),
                cast(vf.JsonData, hinted_entry),
                cast(vf.JsonData | None, ground_truth),
            )
            if is_multi_turn(test_category):
                max_steps = maximum_step_limit()
                row["max_steps_per_turn"] = max_steps
                row["max_turns"] = (
                    len(cast(Sequence[BFCLRawTurn], row["question"])) * max_steps
                )
            else:
                row["max_turns"] = 1
            rows.append(row)
        return rows

    return factory


def load_tasks(test_category: str = "simple_python", examples_per_category: int = -1):
    return build_task_loader(test_category, examples_per_category)()


def bfcl_row(
    test_category: str,
    entry: vf.JsonData,
    hinted_entry: vf.JsonData,
    ground_truth: vf.JsonData | None,
) -> vf.JsonData:
    question = cast(list[BFCLRawTurn], entry["question"])
    first_turn_system_prompt, first_turn_prompt = split_system_prompt(
        normalize_turn(question[0])
    )
    row: vf.JsonData = {
        "task_id": str(entry["id"]),
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
    for key in ("initial_config", "involved_classes"):
        if key in entry:
            row[key] = entry[key]
    if "missed_function" in entry:
        row["missed_function"] = entry["missed_function"]
    if "missed_function" in hinted_entry:
        row["missed_function_with_hints"] = hinted_entry["missed_function"]
    if ground_truth is not None:
        row.update(ground_truth)
    return row


def normalize_turn(value: object) -> list[vf.JsonData]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"role": "user", "content": value}]
    if isinstance(value, dict):
        return [dict(cast(vf.JsonData, value))]
    if isinstance(value, Sequence):
        messages: list[vf.JsonData] = []
        for item in value:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                messages.append(dict(cast(vf.JsonData, item)))
            else:
                raise TypeError(f"Unsupported BFCL message item: {type(item).__name__}")
        return messages
    raise TypeError(f"Unsupported BFCL prompt turn: {type(value).__name__}")


def split_system_prompt(
    messages: Sequence[vf.JsonData],
) -> tuple[list[vf.JsonData], list[vf.JsonData]]:
    system_prompt: list[vf.JsonData] = []
    prompt: list[vf.JsonData] = []
    for message in messages:
        target = system_prompt if message.get("role") == "system" else prompt
        target.append(dict(message))
    return system_prompt, prompt


def maximum_step_limit() -> int:
    patch_bfcl_eval()
    from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT

    return cast(int, MAXIMUM_STEP_LIMIT)


def model_name(state: vf.State) -> str:
    value = state.metadata.get("model")
    return value if isinstance(value, str) and value else "unknown"


def assistant_tool_calls(state: vf.State) -> list[ToolCall]:
    messages = [message for message in state.completion if message.role == "assistant"]
    if not messages:
        return []
    return parse_tool_calls(messages[-1])


def transcript_completion_messages(state: vf.State) -> Messages:
    if not state.transcript:
        return []
    seen = list(state.transcript[0].prompt)
    messages: Messages = []
    for index, turn in enumerate(state.transcript):
        if index:
            prompt_delta = list(turn.prompt[len(seen) :])
            messages.extend(prompt_delta)
            seen.extend(prompt_delta)
        messages.extend(turn.completion)
        seen.extend(turn.completion)
        messages.extend(turn.tool_results)
        seen.extend(turn.tool_results)
    return messages


def parse_tool_calls(message: Message | vf.JsonData) -> list[ToolCall]:
    if isinstance(message, AssistantMessage):
        return list(message.tool_calls or [])
    raw_tool_calls: object
    if isinstance(message, dict):
        raw_tool_calls = message.get("tool_calls") or []
    else:
        raw_tool_calls = getattr(message, "tool_calls", []) or []
    if not isinstance(raw_tool_calls, Sequence):
        return []
    calls: list[ToolCall] = []
    for raw_call in raw_tool_calls:
        if isinstance(raw_call, ToolCall):
            calls.append(raw_call)
            continue
        if not isinstance(raw_call, dict):
            continue
        raw_call = cast(vf.JsonData, raw_call)
        function = raw_call.get("function")
        if isinstance(function, dict):
            function_map = cast(vf.JsonData, function)
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


def convert_to_gorilla(tool_calls: list[ToolCall]) -> list[vf.JsonData]:
    return [
        {tool_call.name: tool_args(tool_call.arguments)} for tool_call in tool_calls
    ]


def convert_to_func_calls(tool_calls: list[ToolCall]) -> list[str]:
    func_calls: list[str] = []
    for tool_call in tool_calls:
        params = tool_args(tool_call.arguments)
        args = ",".join(f"{key}={value!r}" for key, value in params.items())
        func_calls.append(f"{tool_call.name}({args})")
    return func_calls


def json_clone(value: object) -> object:
    return json.loads(json.dumps(value))


def bfcl_involved_classes(task: BFCLTask) -> list[str]:
    value = json_clone(task.involved_classes)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("BFCL multi-turn tasks require involved_classes.")
    return value


def tool_args(value: str) -> vf.JsonData:
    parsed = json.loads(value or "{}")
    if not isinstance(parsed, dict):
        raise TypeError("BFCL tool arguments must decode to an object.")
    return cast(vf.JsonData, parsed)


def relevance_reward(task: vf.Task, state: vf.State) -> float:
    patch_bfcl_eval()
    from bfcl_eval.utils import is_empty_output

    task = cast(BFCLTask, task)
    category = task.category
    try:
        gorilla_tool_calls = convert_to_gorilla(assistant_tool_calls(state))
        contain_func_call = not is_empty_output(gorilla_tool_calls)
    except Exception:
        contain_func_call = False
    if "irrelevance" in category:
        return float(not contain_func_call)
    return float(contain_func_call)


def ast_reward(task: vf.Task, state: vf.State) -> float:
    patch_bfcl_eval()
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.utils import (
        is_function_calling_format_output,
        is_java,
        is_js,
    )

    task = cast(BFCLTask, task)
    category = task.category
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
        task.function,
        gorilla_tool_calls,
        task.ground_truth,
        language,
        category,
        model_name(state),
    )
    return float(bool(checker_result["valid"]))


def multi_turn_reward(task: vf.Task, state: vf.State) -> float:
    patch_bfcl_eval()
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )
    from bfcl_eval.model_handler.base_handler import is_empty_execute_response

    task = cast(BFCLTask, task)
    completion = transcript_completion_messages(state)
    raw_ground_truth = task.ground_truth
    if not isinstance(raw_ground_truth, Sequence):
        return 0.0
    all_ground_truth = cast(list[list[str]], raw_ground_truth)
    all_func_calls: list[list[list[str]]] = [[]]
    try:
        for message in completion:
            role = message.role
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
            "initial_config": task.initial_config,
            "involved_classes": task.involved_classes,
            "id": task.task_id,
        },
        task.task_id.rsplit("_", 1)[0],
        model_name(state),
    )
    return float(bool(result["valid"]))


class BFCLTaskset(vf.Taskset[BFCLTasksetConfig]):
    task_type = BFCLTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        _ = split
        return load_tasks(self.config.test_category, self.config.examples_per_category)

    @vf.reward(weight=1.0)
    async def bfcl_reward(self, task: vf.Task, state: vf.State) -> float:
        patch_bfcl_eval()
        from bfcl_eval.utils import is_multi_turn, is_relevance_or_irrelevance

        task = cast(BFCLTask, task)
        category = task.category
        if is_relevance_or_irrelevance(category):
            return relevance_reward(task, state)
        if is_multi_turn(category):
            return multi_turn_reward(task, state)
        return ast_reward(task, state)


class BFCLHarness(vf.Harness[BFCLHarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = BFCLTask.model_validate(context.task.model_dump())
        state = context.state
        patch_bfcl_eval()
        from bfcl_eval.utils import is_multi_turn

        state.metadata["model"] = context.model
        if is_multi_turn(task.category):
            await self.run_multi_turn(context, task, state)
            return
        prompt = self.initial_messages(task)
        start = time.time()
        response = await context.model_client.get_response(
            prompt=prompt,
            model=context.model,
            sampling_args=self.sampling_args(task, context.sampling_args),
            tools=bfcl_tool_defs(bfcl_functions(task)),
            state=state,
        )
        end = time.time()
        turn = vf.Turn(
            prompt=prompt,
            completion=await parse_response_message(response),
            tool_calls=list(response.message.tool_calls or []),
            response_id=response.id,
            model=response.model,
            created=response.created,
            finish_reason=response.message.finish_reason,
            usage=vf.TurnUsage.from_usage(response.usage),
            tokens=vf.TurnTokens.from_response(
                response.message.tokens,
                is_truncated=bool(response.message.is_truncated),
            ),
            is_truncated=bool(response.message.is_truncated),
            timing=vf.TimeSpan(start=start, end=end),
        )
        state.transcript.append(turn)
        if turn.is_truncated:
            state.is_truncated = True
        state.stop("assistant_completed")

    async def run_multi_turn(
        self,
        context: vf.Context,
        task: BFCLTask,
        state: vf.State,
    ) -> None:
        from bfcl_eval.constants.default_prompts import (
            DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
        )
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )
        from bfcl_eval.model_handler.base_handler import is_empty_execute_response

        messages = list(self.initial_messages(task))
        next_prompts = list(task.question)[1:]
        holdout_function = bfcl_missed_function(task)
        tool_defs = bfcl_tool_defs(bfcl_functions(task))
        initial_config = cast(vf.JsonData, json_clone(task.initial_config or {}))
        involved_classes = bfcl_involved_classes(task)
        simulator_model = (
            model_name(state).replace("/", "_").replace("-", "_").replace(".", "_")
        )
        long_context = "long_context" in task.category or "composite" in task.category
        max_steps_per_turn = int(task.max_steps_per_turn or maximum_step_limit())
        turn_idx = 0
        steps_per_turn = 0
        execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            simulator_model,
            task.task_id,
            long_context=long_context,
        )
        while True:
            response = await context.model_client.get_response(
                prompt=messages,
                model=context.model,
                sampling_args=self.sampling_args(task, context.sampling_args),
                tools=tool_defs,
                state=state,
            )
            turn = vf.Turn(
                prompt=list(messages),
                completion=await parse_response_message(response),
                tool_calls=list(response.message.tool_calls or []),
                response_id=response.id,
                model=response.model,
                created=response.created,
                finish_reason=response.message.finish_reason,
                usage=vf.TurnUsage.from_usage(response.usage),
                tokens=vf.TurnTokens.from_response(
                    response.message.tokens,
                    is_truncated=bool(response.message.is_truncated),
                ),
                is_truncated=bool(response.message.is_truncated),
            )
            state.transcript.append(turn)
            if turn.is_truncated:
                state.is_truncated = True
            messages.extend(turn.completion)
            tool_calls = list(turn.tool_calls)
            try:
                func_calls = convert_to_func_calls(tool_calls)
                if is_empty_execute_response(func_calls):
                    func_calls = []
            except Exception:
                func_calls = []
            if func_calls:
                execution_results, _ = execute_multi_turn_func_call(
                    func_call_list=func_calls,
                    initial_config=initial_config,
                    involved_classes=involved_classes,
                    model_name=simulator_model,
                    test_entry_id=task.task_id,
                    long_context=long_context,
                )
                tool_messages = [
                    ToolMessage(
                        tool_call_id=tool_call.id,
                        content=cast(MessageContent, execution_result),
                    )
                    for execution_result, tool_call in zip(
                        execution_results, tool_calls
                    )
                ]
                turn.tool_results = tool_messages
                messages.extend(tool_messages)
                steps_per_turn += 1
                if steps_per_turn >= max_steps_per_turn:
                    state.stop("max_steps_per_turn_reached")
                    return
                continue

            steps_per_turn = 0
            turn_idx += 1
            if not next_prompts:
                state.stop("no_next_prompt_and_no_tool_calls")
                return
            next_prompt = normalize_turn(next_prompts.pop(0))
            if str(turn_idx) in holdout_function:
                tool_defs.extend(bfcl_tool_defs(holdout_function[str(turn_idx)]))
                if next_prompt:
                    raise ValueError(
                        "BFCL holdout turns must not include user messages."
                    )
                messages.append(
                    UserMessage(content=DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC)
                )
            else:
                for message in next_prompt:
                    role = message.get("role")
                    if role == "assistant":
                        messages.append(AssistantMessage.model_validate(message))
                    elif role == "user":
                        messages.append(UserMessage.model_validate(message))
                    elif role == "tool":
                        messages.append(ToolMessage.model_validate(message))
                    elif role == "system":
                        raise ValueError(
                            "BFCL turn prompts must not include system messages."
                        )
                    else:
                        raise ValueError(
                            f"Unsupported BFCL prompt message role: {role!r}."
                        )


def load_taskset(config: BFCLTasksetConfig) -> BFCLTaskset:
    return BFCLTaskset(config=config)


def load_harness(config: BFCLHarnessConfig) -> BFCLHarness:
    return BFCLHarness(config=config)
