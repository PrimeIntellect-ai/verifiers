"""
Prompt templates for GEPA optimization in Verifiers.

This module contains specialized templates for different component types
(tool descriptions, system prompts, etc.) used during GEPA's reflection phase.
"""

# Tool-specific prompt template for GEPA reflection
TOOL_DESCRIPTION_PROMPT_TEMPLATE = """You are improving the description of a tool (function) that an AI assistant can call.

TOOL NAME: <tool_name>

TOOL PARAMETERS:
```json
<tool_parameters>
```

CURRENT DESCRIPTION:
```
<curr_instructions>
```

The following are examples of how the assistant used this tool, along with feedback on the results:
```
<inputs_outputs_feedback>
```

Your task is to write an improved TOOL DESCRIPTION for the "<tool_name>" tool.

A good tool description should:
- Clearly explain what the tool does and when to use it
- Match the parameter schema shown above
- Mention any important constraints, edge cases, or common mistakes
- Be concise but informative enough for the AI to decide when/how to call this tool

Based on the feedback, identify patterns in tool misuse and improve the description to prevent them.

Provide the new tool description within ``` blocks."""


__all__ = ["TOOL_DESCRIPTION_PROMPT_TEMPLATE"]
