import inspect
import json
from typing import Any, Callable, Dict, List, Tuple

from verifiers import (
    Messages,
    MultiTurnEnv,
    RewardFunc,
    Rubric,
    State,
    XMLParser,
)
from verifiers.utils.data_utils import load_example_dataset


def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()

    # Extract examples if present
    examples = []
    return_description = ""
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
        elif part.startswith("Returns:"):
            return_description = part.split("\n")[1].strip()

    return_type = str(
        sig.return_annotation.__name__
        if sig.return_annotation != inspect.Parameter.empty
        else "any"
    )

    print(f"return_description: {return_description} ({return_type})")
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name) + 1 :].strip()

        args[name] = {
            "type": str(
                param.annotation.__name__
                if param.annotation != inspect.Parameter.empty
                else "any"
            ),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default

    return {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": return_description + f" ({return_type})",
        "examples": examples,
    }


def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]

        desc.append("\nArguments:")
        for arg_name, arg_info in schema["args"].items():
            default = (
                f" (default: {arg_info['default']})" if "default" in arg_info else ""
            )
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")

        if schema["examples"]:
            desc.append("\nExamples:")
            for example in schema["examples"]:
                desc.append(f"  {example}")

        if schema["returns"]:
            desc.append(f"\nReturns: {schema['returns']}")

        descriptions.append("\n".join(desc))

    return "\n\n".join(descriptions)


class XMLToolRubric(Rubric):
    def __init__(
        self,
        parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
        env_parser: XMLParser = XMLParser(fields=["result"]),
        tools: List[Callable] = [],
    ):
        super().__init__(parser=parser)
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {
            tool.__name__ if hasattr(tool, "__name__") else str(tool): tool
            for tool in tools
        }
        self.reward_funcs = [
            self.correct_answer_reward_func,
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
        ]
        self.reward_weights = [
            1.0,
            0.2,
            0.2,
        ]
        for tool_name in self.tools.keys():
            self.reward_funcs.append(self.get_named_tool_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_count_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_attempt_reward_func(tool_name))
            self.reward_weights.append(0.0)

    def evaluate_code(self, code_str, answer, **kwargs) -> float:
        import io
        import signal
        import sys
        from contextlib import redirect_stdout

        try:
            test_cases = json.loads(answer)["test_cases"]
        except Exception:
            return 0.0
        # strip ```python and ``` if present at the beginning and end of the code
        code_str = code_str.strip()
        if code_str.startswith("```python"):
            code_str = code_str[9:]
        elif code_str.startswith("```"):
            code_str = code_str[3:]
        if code_str.endswith("```"):
            code_str = code_str[:-3]
        code_str = code_str.strip()

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        def normalize_output(output):
            # Normalize line endings and whitespace
            return "\n".join(line.strip() for line in output.splitlines())

        total_cases = 0
        passed = 0

        for test in test_cases:
            output = io.StringIO()
            sys.stdin = io.StringIO(test["input"])
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                with redirect_stdout(output):
                    exec(code_str)
                signal.alarm(0)
                actual = normalize_output(output.getvalue())
                expected = normalize_output(test["output"])

                # Compare each line individually
                actual_lines = actual.splitlines()
                expected_lines = expected.splitlines()
                total_cases += len(expected_lines)
                for a, e in zip(actual_lines, expected_lines):
                    if a == e:
                        passed += 1

            except Exception as e:
                sys.stdin = sys.__stdin__
                return 0.0
            sys.stdin = sys.__stdin__

        return passed / total_cases if total_cases else 0.0

    def correct_answer_reward_func(self, completion, answer, **kwargs) -> float:
        """Reward function that checks if the final answer matches the expected answer."""
        response = str(self.parser.parse_answer(completion))
        return 1.0 if answer == response else 0.0

    def tool_execution_reward_func(
        self, completion: List[Dict[str, str]], **kwargs
    ) -> float:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        tool_attempts = 0
        successful_executions = 0

        # Find assistant messages with tools and their responses
        for i, msg in enumerate(completion):
            if msg["role"] == "assistant":
                # Use parser to check for tool tag
                parsed = self.parser.parse(msg["content"])
                if hasattr(parsed, "tool") and parsed.tool is not None:
                    # Found a properly formatted tool message
                    if i + 1 < len(completion) and completion[i + 1]["role"] == "user":
                        tool_attempts += 1
                        # Check response with env_parser
                        parsed_response = self.env_parser.parse(
                            completion[i + 1]["content"]
                        )
                        if (
                            hasattr(parsed_response, "result")
                            and parsed_response.result is not None
                            and not parsed_response.result.startswith("Error:")
                        ):
                            successful_executions += 1

        # Calculate reward
        if tool_attempts == 0:
            return 0.0
        return successful_executions / tool_attempts

    def get_named_tool_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that checks tool execution success for a specific tool.

        Uses XMLParser to identify proper tool calls.
        """

        def tool_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            """
            Reward function that checks execution success for the {tool_name} tool.

            Uses XMLParser to identify proper tool calls for the specified tool.
            """
            import json

            tool_attempts = 0
            successful_executions = 0

            # Find assistant messages with the specific tool and their responses
            for i, msg in enumerate(completion):
                if msg["role"] == "assistant":
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "tool") and parsed.tool is not None:
                        try:
                            command = json.loads(parsed.tool)
                            if (
                                isinstance(command, dict)
                                and command.get("name") == tool_name
                            ):
                                # Found a properly formatted tool message for the specific tool
                                if (
                                    i + 1 < len(completion)
                                    and completion[i + 1]["role"] == "user"
                                ):
                                    tool_attempts += 1
                                    # Check response with env_parser
                                    parsed_response = self.env_parser.parse(
                                        completion[i + 1]["content"]
                                    )
                                    if (
                                        hasattr(parsed_response, "result")
                                        and parsed_response.result is not None
                                        and not parsed_response.result.startswith(
                                            "Error:"
                                        )
                                    ):
                                        successful_executions += 1
                        except json.JSONDecodeError:
                            pass

            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return successful_executions / tool_attempts

        # Create a function with the dynamic name based on tool_name
        tool_reward_func.__name__ = f"{tool_name}_reward_func"
        return tool_reward_func

    def get_named_tool_count_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """

        def tool_count_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            successful_executions = 0.0
            for i, msg in enumerate(completion):
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "tool") and parsed.tool is not None:
                        try:
                            command = json.loads(parsed.tool)
                            if (
                                isinstance(command, dict)
                                and command.get("name") == tool_name
                            ):
                                # Found a properly formatted tool message for the specific tool
                                if (
                                    i + 1 < len(completion)
                                    and completion[i + 1]["role"] == "user"
                                ):
                                    parsed_response = self.env_parser.parse(
                                        completion[i + 1]["content"]
                                    )
                                    if (
                                        hasattr(parsed_response, "result")
                                        and parsed_response.result is not None
                                        and not parsed_response.result.startswith(
                                            "Error:"
                                        )
                                    ):
                                        successful_executions += 1
                        except json.JSONDecodeError:
                            pass
            return successful_executions

        tool_count_reward_func.__name__ = f"{tool_name}_count_reward_func"
        return tool_count_reward_func

    def get_named_tool_attempt_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """

        def tool_attempt_reward_func(
            completion: List[Dict[str, str]], **kwargs
        ) -> float:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            attempted_executions = 0.0
            for i, msg in enumerate(completion):
                if msg["role"] == "assistant":
                    parsed = self.parser.parse(msg["content"])
                    if hasattr(parsed, "tool") and parsed.tool is not None:
                        try:
                            command = json.loads(parsed.tool)
                            if (
                                isinstance(command, dict)
                                and command.get("name") == tool_name
                            ):
                                attempted_executions += 1
                        except json.JSONDecodeError:
                            pass
            return attempted_executions

        tool_attempt_reward_func.__name__ = f"{tool_name}_attempt_reward_func"
        return tool_attempt_reward_func


class XMLToolEnv(MultiTurnEnv):
    def __init__(
        self,
        tools: List[Callable] = [],
        system_prompt: str = "",
        format_prompt: bool = True,
        parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
        env_parser: XMLParser = XMLParser(fields=["result"]),
        max_turns: int = 10,
        **kwargs,
    ):
        rubric = XMLToolRubric(tools=tools, parser=parser, env_parser=env_parser)
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}

        if format_prompt:
            tool_descriptions = format_tool_descriptions(self.tool_schemas)
            formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        else:
            formatted_prompt = system_prompt
        super().__init__(
            system_prompt=formatted_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )
        self.env_parser = env_parser

    def get_reward_funcs(self, **kwargs) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self, **kwargs) -> List[float]:
        return self.rubric.get_reward_weights()

    def is_completed(self, messages: Messages, state: State, **kwargs: Any) -> bool:
        return self.parser.parse_answer(messages) is not None

    def call_tool(self, tool_json: str, max_chars: int = 1024, **kwargs) -> str:
        """Call a tool based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return 'Error: Tool command must be a JSON object, e.g. \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''

            tool_name = command.get("name")
            if not tool_name:
                return 'Error: Tool command must specify \'name\', e.g. \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''

            if tool_name not in self.tools:
                return (
                    f"Error: Unknown tool '{tool_name}. "
                    + 'Please format your tool call as \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''
                )

            tool_func = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, str):
                tool_schema = next(
                    (
                        schema["args"]
                        for schema in self.tool_schemas
                        if schema["name"] == tool_name
                    ),
                    None,
                )
                return f"Error: Arguments for {tool_name} must be a JSON object with schema {tool_schema}, not a string."

            # Call the tool function with arguments
            result = tool_func(**tool_args)
            if max_chars > 0 and len(str(result)) > max_chars:
                result = str(result)[:max_chars] + "..."
            return str(result)
        except Exception as e:
            return (
                f"Error: {str(e)}. "
                + 'Please format your tool call as \'{{"name": "tool_name", "args": {{"arg1": "value1", "arg2": "value2"}}}}\''
            )

    def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Tuple[Messages, State]:
        try:
            parsed = self.parser.parse(messages[-1]["content"])  # type: ignore
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, "tool") and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    return [
                        {
                            "role": "user",
                            "content": self.env_parser.format(result=result),
                        }
                    ], state
                else:
                    return [
                        {
                            "role": "user",
                            "content": "Error: Tool execution returned empty output.",
                        }
                    ], state
        except Exception:
            pass
        return [
            {
                "role": "user",
                "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting.",
            }
        ], state


def load_environment(
    dataset_name: str,
    split: str,
    tools: List[Callable] = [],
    system_prompt: str = "",
    format_prompt: bool = True,
    parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
    env_parser: XMLParser = XMLParser(fields=["result"]),
    max_turns: int = 10,
    **kwargs,
) -> XMLToolEnv:
    dataset = load_example_dataset(name=dataset_name, split=split)
    return XMLToolEnv(
        dataset=dataset,
        tools=tools,
        system_prompt=system_prompt,
        format_prompt=format_prompt,
        parser=parser,
        env_parser=env_parser,
        max_turns=max_turns,
        **kwargs,
    )
