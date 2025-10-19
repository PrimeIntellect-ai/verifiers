import re
from types import SimpleNamespace
from typing import Any, Callable

from verifiers.parsers.parser import Parser
from verifiers.types import ChatMessage, Messages


class XMLParser(Parser):
    def __init__(
        self,
        fields: list[str | tuple[str, ...]],
        answer_field: str = "answer",
        extract_fn: Callable[[str], str] = lambda x: x,
    ):
        """
        Initialize the parser with field definitions.

        Each field may be:
          - a string (e.g. "reasoning"): the XML tag is fixed.
          - a tuple of alternatives (e.g. ("code", "answer")): the first element is
            the canonical name used for formatting, and all elements are allowed tags
            when parsing.

        The schema is assumed to have no duplicate names.
        """
        super().__init__(extract_fn=extract_fn)
        # list of (canonical, [alternatives])
        self._fields: list[tuple[str, list[str]]] = []

        self.answer_field = answer_field
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
        if "think" in seen:
            self.logger.warning(
                "You have included the 'think' field in the XMLParser. This should only be used with models which always include <think>...</think> tags but do NOT parse them automatically. "
                "This will cause parsing failures if the model does not include <think>...</think> tags, or if the chat template automatically removes <think>...</think> tags."
                "In particular, you should NOT use this parser configuration with Qwen3 or DeepSeek-R1 models."
            )

    def parse(self, text: str, strip: bool = True, last: bool = False) -> Any:
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
        """
        results: dict[str, str | None] = {}
        for canonical, alternatives in self._fields:
            # For each allowed alternative tag, search independently.
            for alt in alternatives:
                # Regex pattern to capture the content between the tags.
                pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
                if last:
                    match = None
                    for match in re.finditer(pattern, text, re.DOTALL):
                        pass  # iterate over matches to bind last match
                else:
                    match = re.search(pattern, text, re.DOTALL)
                if match:
                    results[alt] = match.group(1).strip() if strip else match.group(1)
                else:
                    results[alt] = None
        return SimpleNamespace(**results)

    def parse_answer(self, completion: Messages) -> str | None:
        """Extract the last answer from a completion."""
        if isinstance(completion, str):
            parsed = self.parse(completion, last=True)
            if (
                parsed
                and hasattr(parsed, self.answer_field)
                and getattr(parsed, self.answer_field) is not None
            ):
                return getattr(parsed, self.answer_field)
        else:
            for msg in reversed(self.get_assistant_messages(completion)):
                assert "content" in msg
                content = str(msg["content"])
                parsed = self.parse(content)
                if (
                    parsed
                    and hasattr(parsed, self.answer_field)
                    and getattr(parsed, self.answer_field) is not None
                ):
                    return getattr(parsed, self.answer_field)
        return None

    def get_format_str(self) -> str:
        """
        Return a string that describes the format of the XML.
        """
        format_str = ""
        for field in self._fields:
            if len(field[1]) > 1:
                options = " | ".join(field[1])
                format_str += f"<[ {options} ]>\n...\n</[ {options} ]>\n"
            else:
                format_str += f"<{field[0]}>\n...\n</{field[0]}>\n"
        return format_str.strip()

    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that checks if messages follow the expected format.

        The function does not make assumptions about which fields should start/end the message
        or the specific order of fields. It checks that:
        - At least one field from the schema is present in each message
        - Fields have proper content and spacing
        """

        def format_reward_func(completion: list[ChatMessage]):
            """Reward function that checks if each step follows the expected format."""
            model_messages = self.get_assistant_messages(completion)
            if not model_messages:
                return 0.0

            # Calculate format adherence for each message
            format_scores = []
            for msg in model_messages:
                assert "content" in msg
                content = str(msg["content"])
                parsed = self.parse(content)
                parsed_no_strip = self.parse(content, strip=False)

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
                            content.count(f"<{alt}>") > 0
                            or content.count(f"</{alt}>") > 0
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
                    if content.strip().startswith(f"<{alt}>"):
                        starts_with_any_field = True
                        break

                # Check if any field from the last field set ends the message
                ends_with_any_field = False
                last_field_set = self._fields[-1][
                    1
                ]  # Get alternatives for last field set
                for alt in last_field_set:
                    if content.strip().endswith(f"</{alt}>"):
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

    def get_fields(self) -> list[str]:
        """Return a list of the canonical field names (in order)."""
        return [canonical for canonical, _ in self._fields]

    def format(self, **kwargs) -> str:
        """
        Format the provided keyword arguments into an XML string.

        For fields with alternatives (tuple), the canonical name (the first element)
        is used as the XML tag. The method looks for a provided value using any of the
        allowed names (preferring the canonical if present).

        Example usage:
            parser = XMLParser(['reasoning', ('code', 'answer')])
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
