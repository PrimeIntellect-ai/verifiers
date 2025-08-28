import json
import re
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import Dataset, concatenate_datasets
from smolagents.default_tools import PythonInterpreterTool  # type: ignore
from smolagents.tools import Tool  # type: ignore

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers.parser import Parser
from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RewardFunc, State
from verifiers.utils.data_utils import load_example_dataset

DEFAULT_SMOLA_PROMPT_TEMPLATE = """You are an intelligent assistant designed to solve problems that require careful reasoning.

When tackling a task, you should:
1. Break the problem down into steps
2. Reason carefully about how to solve it
3. Use available tools to help you solve the problem
4. Provide a clear final answer

Available tools:
{tool_descriptions}

Format your response using these XML tags:
<reasoning>
Think step-by-step about how to solve the task.
</reasoning>

<tool>
{{
  "name": "tool_name",
  "args": {{
    "arg1": "value1",
    "arg2": "value2"
  }}
}}
</tool>

<answer>
Your final answer or response to the user's request.
</answer>

First use the <reasoning> tag to think through the problem. When you need to use a tool, use the <tool> tag with the appropriate JSON format. When you're ready to provide the final answer, use the <answer> tag.
"""

MATH_SMOLA_PROMPT_TEMPLATE = """You are an intelligent math assistant designed to solve math problems that require careful reasoning.

When solving a math problem, you should:
1. Break the problem down into steps
2. Reason carefully through each step
3. Use the calculator tool to help with calculations
4. Provide a clear final answer in simplified form

Available tools:
{tool_descriptions}

Format your response using these XML tags:
<reasoning>
Think step-by-step about how to solve the math problem, explaining the approach clearly.
</reasoning>

<tool>
{{
  "name": "calculator", 
  "args": {{
    "expression": "math expression to calculate"
  }}
}}
</tool>

<answer>
Your final answer to the math problem, in simplified form.
</answer>

First use the <reasoning> tag to think through the problem. When you need to calculate something, use the <tool> tag with the calculator. When you're ready to provide the final answer, use the <answer> tag.
"""


"""
Few shot examples for SmolaAgents tool-based assistants.
"""

CALCULATOR_SMOLA_FEW_SHOTS = [
    {
        "role": "user",
        "content": "Janet has 3 times as many marbles as Tom. If Tom has 12 marbles, how many marbles do they have together?",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
Let's solve this step by step:
1. First, calculate how many marbles Janet has (3 times Tom's)
</reasoning>

<tool>
{
  "name": "calculator", 
  "args": {
    "expression": "3 * 12"
  }
}
</tool>

<result>
36
</result>

<reasoning>
2. Now we know:
   - Tom has 12 marbles
   - Janet has 36 marbles
   Let's add them together
</reasoning>

<tool>
{
  "name": "calculator", 
  "args": {
    "expression": "36 + 12"
  }
}
</tool>

<result>
48
</result>

<reasoning>
Together they have 48 marbles:
- Tom: 12 marbles
- Janet: 36 marbles (3 times Tom's)
- Total: 48 marbles
</reasoning>

<answer>
48 marbles
</answer>""",
    },
    {
        "role": "user",
        "content": "Samantha is baking cookies. Each batch requires 2.5 cups of flour. If she has 10 cups of flour, how many complete batches can she make?",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
To find how many complete batches Samantha can make, I need to divide the total amount of flour by the amount needed per batch.
</reasoning>

<tool>
{
  "name": "calculator", 
  "args": {
    "expression": "10 / 2.5"
  }
}
</tool>

<result>
4.0
</result>

<reasoning>
Samantha has 10 cups of flour and each batch requires 2.5 cups of flour.
10 ÷ 2.5 = 4
So Samantha can make 4 complete batches of cookies with her 10 cups of flour.
</reasoning>

<answer>
4 batches
</answer>""",
    },
]

SEARCH_SMOLA_FEW_SHOTS = [
    {
        "role": "user",
        "content": "Question: Which of the following is an example of mechanical weathering?\\n\\nChoices:\\nA. A rock turns red when iron minerals are exposed to oxygen.\\nB. A rock breaks apart after water freezes and expands in a crack.\\nC. A rock dissolves as acidic rainwater flows over its surface.\\nD. A rock changes composition when exposed to extreme heat.",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
I need to understand what mechanical weathering is and which of these options matches that phenomenon. Let me search for information about mechanical weathering.
</reasoning>

<tool>
{
  "name": "web_search",
  "args": {
    "query": "what is mechanical weathering examples"
  }
}
</tool>

<result>
• Mechanical Weathering - Definition, Processes and Types
  Mechanical weathering, also known as physical weathering, is the process by which rocks are broken down into smaller pieces by external forces. Examples include frost wedging (when water freezes and expands in cracks), thermal expansion/contraction, and pressure release.

• Mechanical and Chemical Weathering - Earth Science
  Mechanical weathering breaks rocks into smaller pieces without changing their composition. Ice wedging occurs when water gets into cracks, freezes, and expands, breaking the rock apart. Other forms include plant root growth, animal activity, and abrasion.

• Mechanical Weathering: Definition, Types, Examples - Geology Science
  Types of mechanical weathering include frost wedging, salt crystallization, thermal expansion, plant activity, and animal activity. The most common mechanical weathering process in cold climates is frost wedging, where water enters rock cracks, freezes, and expands.
</result>

<reasoning>
Based on the search results, mechanical weathering is the process by which rocks are broken down into smaller pieces by external forces, without changing their chemical composition. Some examples include:

1. Frost wedging - when water gets into cracks, freezes, and expands, breaking the rock apart
2. Thermal expansion/contraction
3. Plant root growth
4. Animal activity
5. Abrasion

Looking at the answer choices:
A. A rock turns red when iron minerals are exposed to oxygen - This is chemical weathering, as it involves a chemical change (oxidation).
B. A rock breaks apart after water freezes and expands in a crack - This is frost wedging, a form of mechanical weathering.
C. A rock dissolves as acidic rainwater flows over its surface - This is chemical weathering (dissolution).
D. A rock changes composition when exposed to extreme heat - This is a chemical/thermal change, not mechanical weathering.

The answer is B, as it's the only option that describes mechanical weathering.
</reasoning>

<answer>
B
</answer>""",
    },
]

TOOL_SMOLA_FEW_SHOTS = [
    {"role": "user", "content": "What is the current working directory?"},
    {
        "role": "assistant",
        "content": """<reasoning>
Let's use the pwd command to find out the current working directory.
</reasoning>

<tool>
{
  "name": "python_interpreter",
  "args": {
    "code": "import os; print(os.getcwd())"
  }
}
</tool>

<result>
Stdout:
/Users/user/project
Output: None
</result>

<reasoning>
The current working directory is /Users/user/project.
</reasoning>

<answer>
/Users/user/project
</answer>""",
    },
    {
        "role": "user",
        "content": "How many Python files are in the current directory and its subdirectories?",
    },
    {
        "role": "assistant",
        "content": """<reasoning>
Let's use Python to count the number of Python files in the current directory and its subdirectories.
</reasoning>

<tool>
{
  "name": "python_interpreter",
  "args": {
    "code": "import os; count = 0; \\nfor root, dirs, files in os.walk('.'): \\n    count += sum(1 for file in files if file.endswith('.py')); \\nprint(count)"
  }
}
</tool>

<result>
Stdout:
42
Output: None
</result>

<reasoning>
There are 42 Python files in the current directory and its subdirectories.
</reasoning>

<answer>
42
</answer>""",
    },
]


class CalculatorTool(Tool):
    """A calculator tool for evaluating mathematical expressions."""

    name = "calculator"
    description = "Evaluates a single line of Python math expression. No imports or variables allowed."
    inputs = {
        "expression": {
            "type": "string",
            "description": "A mathematical expression using only numbers and basic operators (+,-,*,/,**,())",
        }
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed = set("0123456789+-*/.() ")
        self.is_initialized = True

    def forward(self, expression: str) -> str:  # type: ignore
        """Evaluates a single line of Python math expression. No imports or variables allowed.

        Args:
            expression: A mathematical expression using only numbers and basic operators (+,-,*,/,**,())

        Returns:
            The result of the calculation or an error message

        Examples:
            "2 + 2" -> "4"
            "3 * (17 + 4)" -> "63"
            "100 / 5" -> "20.0"
        """
        if not all(c in self.allowed for c in expression):
            return "Error: Invalid characters in expression"

        try:
            # Safely evaluate the expression with no access to builtins
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class SmolagentsParser(Parser):
    def __init__(self, fields: List[Union[str, Tuple[str, ...]]]):
        """
        Initialize the parser with field definitions.

        Each field may be:
          - a string (e.g. "reasoning"): the XML tag is fixed.
          - a tuple of alternatives (e.g. ("code", "answer")): the first element is
            the canonical name used for formatting, and all elements are allowed tags
            when parsing.

        The schema is assumed to have no duplicate names.
        """
        self._fields: List[
            Tuple[str, List[str]]
        ] = []  # List of (canonical, [alternatives])
        seen = set()
        for field in fields:
            if isinstance(field, str):
                canonical = field
                alternatives = [field]
            elif isinstance(field, tuple):
                if not field:
                    raise ValueError("Field tuple cannot be empty.")
                canonical = field[0]
                if not all(isinstance(alt, str) for alt in field):
                    raise TypeError("All alternatives in a tuple must be strings.")
                alternatives = list(field)
            else:
                raise TypeError("Each field must be a string or a tuple of strings.")
            if canonical in seen:
                raise ValueError(f"Duplicate field name: {canonical}")
            seen.add(canonical)
            self._fields.append((canonical, alternatives))

    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that checks if messages follow the expected format.

        The function does not make assumptions about which fields should start/end the message
        or the specific order of fields. It checks that:
        - At least one field from the schema is present in each message
        - Fields have proper content and spacing
        """

        def format_reward_func(completion, **kwargs) -> float:
            """
            Reward function that checks if each step follows the expected format.

            Args:
                completion: List[Dict[str, str]] | str
                **kwargs: additional kwargs

            Returns:
                float: Score between 0 and 1.
            """
            model_messages = self.get_assistant_messages(completion)
            if not model_messages:
                return 0.0

            # Calculate format adherence for each message
            format_scores = []
            for msg in model_messages:
                content = msg["content"]  # type: ignore
                parsed = self.parse(content)  # type: ignore
                parsed_no_strip = self.parse(content, strip=False)  # type: ignore

                # Check if the message has at least one valid field
                has_any_field = False
                fields_with_content = 0
                total_fields = 0

                # Keep track of which expected fields are present
                expected_field_count = len(
                    self._fields
                )  # Total number of expected field sets
                present_field_sets = (
                    set()
                )  # Which field sets have at least one alternative present

                # Check proper spacing for fields
                has_correct_spacing = True

                for i, (canonical, alternatives) in enumerate(self._fields):
                    field_set_present = False
                    for alt in alternatives:
                        if hasattr(parsed, alt) and getattr(parsed, alt) is not None:
                            has_any_field = True
                            fields_with_content += 1
                            total_fields += 1
                            field_set_present = True

                            # Check if field exists in non-stripped version too (proper spacing)
                            if not (
                                hasattr(parsed_no_strip, alt)
                                and getattr(parsed_no_strip, alt) is not None
                            ):
                                has_correct_spacing = False
                        elif (
                            content.count(f"<{alt}>") > 0  # type: ignore
                            or content.count(f"</{alt}>") > 0  # type: ignore
                        ):
                            # Tag exists but content wasn't properly parsed
                            total_fields += 1
                            field_set_present = True

                    # If any alternative from this field set was present, count it
                    if field_set_present:
                        present_field_sets.add(i)

                # Calculate format score components
                format_score = 0.0

                # Check if any field from the first field set starts the message
                starts_with_any_field = False
                first_field_set = self._fields[0][
                    1
                ]  # Get alternatives for first field set
                for alt in first_field_set:
                    if content.strip().startswith(f"<{alt}>"):  # type: ignore
                        starts_with_any_field = True
                        break

                # Check if any field from the last field set ends the message
                ends_with_any_field = False
                last_field_set = self._fields[-1][
                    1
                ]  # Get alternatives for last field set
                for alt in last_field_set:
                    if content.strip().endswith(f"</{alt}>"):  # type: ignore
                        ends_with_any_field = True
                        break

                # Weight the score based on different criteria
                if has_any_field:
                    # Calculate the proportion of expected field sets that are present
                    field_set_ratio = len(present_field_sets) / expected_field_count
                    format_score += 0.4 * field_set_ratio

                if has_correct_spacing:
                    format_score += 0.2

                if starts_with_any_field:
                    format_score += 0.2

                if ends_with_any_field:
                    format_score += 0.2

                format_scores.append(format_score)

            # Return average format adherence
            if not format_scores:
                return 0.0
            return sum(format_scores) / len(format_scores)

        return format_reward_func

    def get_fields(self) -> List[str]:
        """Return a list of the canonical field names (in order)."""
        return [canonical for canonical, _ in self._fields]

    def format(self, **kwargs) -> str:
        """
        Format the provided keyword arguments into an XML string.

        For fields with alternatives (tuple), the canonical name (the first element)
        is used as the XML tag. The method looks for a provided value using any of the
        allowed names (preferring the canonical if present).

        Example usage:
            parser = SmolaParser(['reasoning', ('code', 'answer')])
            formatted_str = parser.format(reasoning="...", code="...")
        """
        parts = []
        for canonical, alternatives in self._fields:
            value = None
            # Look for a provided value using any of the acceptable keys,
            # preferring the canonical name if it exists.
            if canonical in kwargs:
                value = kwargs[canonical]
            else:
                for alt in alternatives:
                    if alt in kwargs:
                        value = kwargs[alt]
                        break
            if value is None:
                raise ValueError(
                    f"Missing value for field '{canonical}' (allowed: {alternatives})"
                )
            # Use the canonical name as the tag for formatting.
            parts.append(f"<{canonical}>\n{value}\n</{canonical}>")
        return "\n".join(parts)

    def parse(self, text: str, strip: bool = True) -> Any:
        """
        Parse the given XML string and return an object with attributes corresponding
        to all allowed tags in the schema.

        For each field defined:
          - If it is a simple field (e.g. 'reasoning'), the output object will have
            an attribute 'reasoning' set to the text content (or None if missing).
          - If it is defined with alternatives (e.g. ("code", "answer")), the output
            object will have attributes for *each* allowed tag name. For example,
            if the schema is ['reasoning', ('code', 'answer')], then both
            `result.code` and `result.answer` are always accessible. If a tag is not
            found in the XML, its corresponding attribute is set to None.

        This implementation also attempts to parse tool JSON content when available.
        """
        results: Dict[str, Optional[str]] = {}
        for canonical, alternatives in self._fields:
            # For each allowed alternative tag, search independently.
            for alt in alternatives:
                # Regex pattern to capture the content between the tags.
                pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    content = match.group(1).strip() if strip else match.group(1)
                    # If the field contains valid JSON and it's a tool field, parse it
                    if alt == "tool" and content:
                        try:
                            # Try to parse as JSON but preserve the string for the result
                            json.loads(content)
                        except json.JSONDecodeError:
                            pass
                    results[alt] = content
                else:
                    results[alt] = None
        return SimpleNamespace(**results)


class LegacyToolRubric(Rubric):
    def __init__(
        self,
        parser: Parser = XMLParser(fields=["reasoning", ("tool", "answer")]),
        env_parser: Parser = XMLParser(fields=["result"]),
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


class SmolagentsToolRubric(LegacyToolRubric):
    def __init__(
        self,
        parser: SmolagentsParser = SmolagentsParser(
            fields=["reasoning", ("tool", "answer")]
        ),
        env_parser: SmolagentsParser = SmolagentsParser(fields=["result"]),
        tools: List[Any] = [],
    ):
        super().__init__()
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {tool.name: tool for tool in tools}
        self.reward_funcs = [
            self.parser.get_format_reward_func(),
        ]
        self.reward_weights = [
            1.0,
            0.2,
        ]
        for tool_name in self.tools.keys():
            self.add_reward_func(self.get_named_tool_reward_func(tool_name), weight=0.0)

    def evaluate_code(self, code_str, answer, **kwargs) -> float:
        import io
        import signal
        from contextlib import redirect_stdout

        try:
            test_cases = json.loads(answer)["test_cases"]
        except Exception as e:
            print(f"Error parsing test cases: {e}")
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


class SmolagentsToolEnv(MultiTurnEnv):
    def __init__(
        self,
        dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        tools: List[Any] = [],
        system_prompt: str = "",
        few_shot: List[Dict[str, str]] = [],
        mask_env_response: bool = True,
        max_steps: int = 10,
        **kwargs,
    ):
        # Format the system prompt with tool descriptions
        tool_descriptions = self._format_tool_descriptions(tools)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            **kwargs,
        )
        self.dataset_name = dataset
        self.max_steps = max_steps
        self.tools = {tool.name: tool for tool in tools}
        self.rubric = SmolagentsToolRubric(tools=tools)
        self.llm_parser = SmolagentsParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = SmolagentsParser(fields=["result"])

    def _format_tool_descriptions(self, tools: List[Any]) -> str:
        """Formats tool schemas into a user-friendly description string."""
        descriptions = []
        for tool in tools:
            desc = [f"{tool.name}: {tool.description}"]

            desc.append("\nArguments:")
            for arg_name, arg_info in tool.inputs.items():
                desc.append(f"  - {arg_name}: {arg_info['description']}")

            desc.append(f"\nReturns: {tool.output_type}")

            descriptions.append("\n".join(desc))

        return "\n\n".join(descriptions)

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()

    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def _get_step_count(self, messages: Messages) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        assert isinstance(messages, list)
        step_count = 0

        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)

        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
        return step_count

    def is_completed(self, messages: Messages, state: State, **kwargs: Any) -> bool:
        assert isinstance(messages, list)
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count > self.max_steps:
                return True

            parsed = self.llm_parser.parse(messages[-1]["content"])  # type: ignore
            # Check if we got a valid answer field (not just None from failed parsing)
            return hasattr(parsed, "answer") and parsed.answer is not None
        except Exception:
            return False

    def call_tool(self, tool_json: str, **kwargs: Any) -> str:
        """Call a Smolagents Tool object based on JSON command."""
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return 'Error: Tool command must be a JSON object, e.g. \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''

            tool_name = command.get("name")
            if not tool_name:
                return 'Error: Tool command must specify \'name\', e.g. \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''

            if tool_name not in self.tools:
                return (
                    f"Error: Unknown tool '{tool_name}'. "
                    + 'Please format your tool call as \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''
                )

            tool = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, str):
                return f"Error: Arguments for {tool_name} must be a JSON object matching the tool's input schema, not a string."

            # Call the tool object with arguments
            result = tool(**tool_args)
            return str(result)
        except json.JSONDecodeError:
            return 'Error: Invalid JSON format. Please format your tool call as \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''
        except Exception as e:
            return (
                f"Error: {str(e)}. "
                + 'Please format your tool call as \'{"name": "tool_name", "args": {"arg1": "value1", "arg2": "value2"}}\''
            )

    def env_response(
        self, messages: Messages, state: State, **kwargs: Any
    ) -> Tuple[Messages, State]:
        assert isinstance(messages, list)
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])  # type: ignore
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, "tool") and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    return [
                        {
                            "role": "user",
                            "content": self.env_parser.format(result=result),
                        }
                    ], {}
                else:
                    return [
                        {
                            "role": "user",
                            "content": "Error: Tool execution returned empty output.",
                        }
                    ], {}
        except Exception:
            pass
        return [
            {
                "role": "user",
                "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting.",
            }
        ], {}


def load_environment(use_few_shot: bool = False, **kwargs):
    dataset = load_example_dataset("math", "train", n=6000)

    eval_aime24 = load_example_dataset("aime2024", n=30)
    eval_aime25 = load_example_dataset("aime2025", n=30)
    eval_dataset = concatenate_datasets([eval_aime24, eval_aime25]).shuffle(seed=0)

    # Use SmolaAgents' PythonInterpreterTool as a replacement for the python tool
    python_tool = PythonInterpreterTool(authorized_imports=["math", "sympy", "numpy"])
    # Add our custom calculator tool
    calculator_tool = CalculatorTool()

    vf_env = SmolagentsToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=MATH_SMOLA_PROMPT_TEMPLATE,
        few_shot=CALCULATOR_SMOLA_FEW_SHOTS if use_few_shot else [],
        tools=[python_tool, calculator_tool],
        max_steps=5,
        **kwargs,
    )

    return vf_env
