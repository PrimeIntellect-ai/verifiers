"""
Test RLM Sub-LLM Calls Environment.

This environment is designed to explicitly test the sub-LLM call functionality
in RLMEnv. It presents a multi-section summarization task that CANNOT be solved with
pure Python code - the model MUST use llm_batch() calls for semantic understanding.

The task:
1. Context contains multiple distinct text sections (articles/paragraphs)
2. Model must summarize each section using llm_batch() calls
3. Summaries are combined into a final structured answer
4. Tests parallel execution via llm_batch()
"""

from datasets import Dataset

import verifiers as vf
from verifiers.envs.rlm_env import RLMEnv


# Sample sections that require semantic understanding to summarize
# These are intentionally complex enough that pure Python string manipulation won't work
_SAMPLE_SECTIONS = [
    {
        "id": "tech",
        "title": "Advances in Quantum Computing",
        "content": """Recent breakthroughs in quantum error correction have brought us closer to 
practical quantum computers. Researchers at leading institutions have demonstrated 
that by using topological qubits and advanced error-correcting codes, quantum systems 
can maintain coherence for significantly longer periods. This development addresses 
one of the fundamental challenges in quantum computing: the fragility of quantum states. 
The implications for cryptography, drug discovery, and optimization problems are profound, 
though commercial applications may still be several years away.""",
    },
    {
        "id": "health",
        "title": "New Insights into Sleep and Memory",
        "content": """A comprehensive study involving over 10,000 participants has revealed 
surprising connections between sleep patterns and memory consolidation. The research 
shows that the quality of deep sleep, rather than total sleep duration, is the critical 
factor for transferring short-term memories to long-term storage. Participants who 
achieved consistent deep sleep cycles performed 40% better on memory recall tests. 
The findings suggest that sleep optimization techniques focusing on deep sleep phases 
could be more effective than simply extending sleep duration.""",
    },
    {
        "id": "environment",
        "title": "Ocean Carbon Sequestration Potential",
        "content": """Marine biologists have discovered that certain species of deep-sea 
phytoplankton are far more efficient at carbon sequestration than previously thought. 
These organisms, found at depths of 200-500 meters, can capture and store carbon 
dioxide at rates up to three times higher than surface phytoplankton. The discovery 
opens new possibilities for natural carbon capture strategies, though scientists 
caution that any intervention in ocean ecosystems must be carefully studied to avoid 
unintended consequences for marine food webs.""",
    },
    {
        "id": "economics",
        "title": "The Rise of Digital Currencies in Emerging Markets",
        "content": """Emerging economies are increasingly adopting digital currencies as 
alternatives to traditional banking systems. In regions with limited banking 
infrastructure, mobile-based digital currencies have enabled millions of previously 
unbanked individuals to participate in the formal economy. Transaction costs have 
dropped by an average of 60%, and financial inclusion rates have improved dramatically. 
However, regulatory challenges and concerns about monetary policy control remain 
significant hurdles for widespread governmental adoption.""",
    },
    {
        "id": "space",
        "title": "Private Space Stations and the Future of Orbital Research",
        "content": """As the International Space Station approaches retirement, private 
companies are racing to develop commercial space stations. These new orbital platforms 
promise to democratize space research by offering laboratory space to universities, 
pharmaceutical companies, and manufacturing firms at a fraction of traditional costs. 
Microgravity environments enable unique material science experiments and biological 
research impossible on Earth. The transition from government to commercial space 
infrastructure marks a fundamental shift in how humanity approaches orbital activities.""",
    },
]


def load_environment(
    num_sections: int = 5,
    num_samples: int = 3,
    max_sub_llm_parallelism: int = 5,
    **kwargs,
) -> vf.Environment:
    """
    Load the test environment for RLM sub-LLM calls.

    Args:
        num_sections: Number of sections to include in each example (1-5)
        num_samples: Number of examples in the dataset
        max_sub_llm_parallelism: Maximum number of concurrent sub-LLM calls
        **kwargs: Additional arguments passed to RLMEnv

    Returns:
        Configured RLMEnv instance
    """
    num_sections = min(num_sections, len(_SAMPLE_SECTIONS))

    dataset_rows = []
    for i in range(num_samples):
        # Select sections for this example
        sections = _SAMPLE_SECTIONS[:num_sections]

        # Build the context as a structured text with clear section markers
        context_parts = []
        for section in sections:
            context_parts.append(
                f"=== {section['title']} ===\n{section['content'].strip()}"
            )
        context = "\n\n".join(context_parts)

        # Store section IDs for reward validation
        section_ids = [s["id"] for s in sections]

        dataset_rows.append(
            {
                "example_id": i,
                "prompt": [
                    {
                        "role": "user",
                        "content": f"""You have {num_sections} text sections in the context. Your task is to:

1. Use the llm_batch() function to summarize EACH section in exactly one sentence
2. You MUST use llm_batch() for summarization - do NOT try to summarize manually
3. llm_batch() takes a list of prompts and returns a list of responses (executed in parallel)
4. Format your final answer as a numbered list of summaries

Example code pattern:
```python
# Get summaries in parallel using llm_batch
summaries = llm_batch([
    f"Summarize this in one sentence: {{section1}}",
    f"Summarize this in one sentence: {{section2}}",
    # ... etc
])

# Build the final answer
answer["content"] = "\\n".join([f"{{i+1}}. {{s}}" for i, s in enumerate(summaries)])
answer["ready"] = True
```""",
                    }
                ],
                "task": "multi-section-summary",
                "answer": "",  # No fixed answer - we check format/content
                "info": {
                    "context": context,
                    "section_ids": section_ids,
                    "num_sections": num_sections,
                },
            }
        )

    dataset = Dataset.from_list(dataset_rows)

    # Reward functions
    def format_reward(state: vf.State) -> float:
        """
        Check if the answer has the correct format (numbered list with N items).
        """
        final_answer = state.get("final_answer", "").strip()
        num_sections = state.get("info", {}).get("num_sections", 5)

        if not final_answer:
            return 0.0

        # Check for numbered items (1. ... 2. ... etc)
        lines = [l.strip() for l in final_answer.split("\n") if l.strip()]
        numbered_lines = [l for l in lines if l and l[0].isdigit() and "." in l[:3]]

        # Reward based on how many sections were summarized
        if len(numbered_lines) >= num_sections:
            return 1.0
        elif len(numbered_lines) > 0:
            return len(numbered_lines) / num_sections
        return 0.0

    def length_reward(state: vf.State) -> float:
        """
        Check that summaries are reasonable length (not too short, not too long).
        Encourages actual summarization rather than copying or one-word answers.
        """
        final_answer = state.get("final_answer", "").strip()

        if not final_answer:
            return 0.0

        lines = [l.strip() for l in final_answer.split("\n") if l.strip()]
        numbered_lines = [l for l in lines if l and l[0].isdigit() and "." in l[:3]]

        if not numbered_lines:
            return 0.0

        # Check each summary is reasonable length (10-200 words)
        good_summaries = 0
        for line in numbered_lines:
            # Remove the number prefix
            content = line.split(".", 1)[-1].strip() if "." in line else line
            word_count = len(content.split())
            if 10 <= word_count <= 200:
                good_summaries += 1

        return good_summaries / len(numbered_lines) if numbered_lines else 0.0

    def completion_reward(state: vf.State) -> float:
        """
        Binary reward for completing the task (has any valid output).
        """
        final_answer = state.get("final_answer", "").strip()
        return 1.0 if len(final_answer) > 50 else 0.0

    rubric = vf.Rubric(
        funcs=[format_reward, length_reward, completion_reward],
        weights=[0.5, 0.3, 0.2],
    )

    env = RLMEnv(
        max_turns=30,
        max_iterations=25,
        timeout_seconds=600.0,
        request_timeout=240.0,
        max_output_length=8192,
        max_sub_llm_parallelism=max_sub_llm_parallelism,
        context_key="context",
        dataset=dataset,
        rubric=rubric,
        interception_host=kwargs.get("interception_host"),
    )

    return env
