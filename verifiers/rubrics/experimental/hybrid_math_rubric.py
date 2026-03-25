from __future__ import annotations

import asyncio

from openai import AsyncOpenAI
from verifiers.envs.experimental.sandbox_mixin import SandboxMixin
from verifiers.parsers.parser import Parser
from verifiers.rubrics.math_rubric import MathRubric
from verifiers.utils.data_utils import extract_boxed_answer

import verifiers as vf

# https://github.com/open-compass/CompassVerifier/blob/2d7cba6df0b21f9c6121786ac1e5770c68473598/src/prompts.py#L28
DEFAULT_JUDGE_PROMPT = """\
As a grading expert, your task is to determine whether the candidate's final answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly.
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) → Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) → Label as REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") → Label as REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{{C}} - INCOMPLETE).

Grading Scale:
\\boxed{{A}} - CORRECT:
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{{B}} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{{C}} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition), or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{{C}} with the corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order, multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs, equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer, for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{{A/B/C}} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{answer}
<Standard Answer End>

<Candidate's Answer Begin>
{response}
<Candidate's Answer End>

Analysis step by step and Final Judgment:
"""


class HybridMathRubric(vf.JudgeRubric):
    """Runs rule-based math verification first, with optional LLM judge fallback.

    Delegates math verification to an internal :class:`MathRubric` instance so
    the executor-based, non-blocking verification logic is reused rather than
    duplicated.
    """

    DEFAULT_JUDGE_PARSER = None
    DEFAULT_JUDGE_MODEL = "gpt-5-nano"
    DEFAULT_JUDGE_CLIENT = None
    DEFAULT_JUDGE_PROMPT = DEFAULT_JUDGE_PROMPT
    DEFAULT_JUDGE_SAMPLING_ARGS = {}
    DEFAULT_USE_JUDGE_FALLBACK = False

    def __init__(
        self,
        parser: Parser | None = DEFAULT_JUDGE_PARSER,
        use_judge_fallback: bool = DEFAULT_USE_JUDGE_FALLBACK,
        judge_client: AsyncOpenAI | None = DEFAULT_JUDGE_CLIENT,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
        judge_sampling_args: dict | None = None,
        timeout_seconds: float = 5,
        max_workers: int = 50,
        **kwargs,
    ):
        judge_sampling_args = judge_sampling_args or self.DEFAULT_JUDGE_SAMPLING_ARGS
        if judge_client is None and not use_judge_fallback:
            judge_client = AsyncOpenAI(api_key="unused")
        super().__init__(
            judge_client=judge_client,
            judge_sampling_args=judge_sampling_args,
            judge_prompt=judge_prompt,
            parser=parser,
            **kwargs,
        )
        # Reward functions
        self.add_reward_func(self.math_verify_score, weight=0)
        self.add_reward_func(self.judge_score, weight=0)
        self.add_reward_func(self.correct_answer, weight=1)

        self.judge_model = judge_model if use_judge_fallback else None
        self.class_objects["judge_model"] = self.judge_model

        # Delegate math verification to default MathRubric
        # We clear its auto-registered reward func since we manage scoring ourselves
        self.math_rubric = MathRubric(
            parser=self.parser,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )
        self.math_rubric.funcs.clear()
        self.math_rubric.weights.clear()

    async def teardown(self):
        await self.math_rubric.teardown()
        await super().teardown()

    async def math_verify_score(
        self, completion: vf.Messages, answer: str, state: vf.State, **kwargs
    ) -> float:
        """Basic rule-based math verification."""
        score = await self.math_rubric.correct_answer(
            parser=self.parser,
            completion=completion,
            answer=answer,
        )
        state["math_verify_score"] = score
        return score

    async def judge_score(
        self,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **kwargs,
    ) -> float:
        """Calls judge if math verification failed and a judge model is set."""
        if state.get("math_verify_score", 0) == 1 or self.judge_model is None:
            return state.get("math_verify_score", 0)

        judge_response = await self.judge(prompt, completion, answer, state)
        judge_result = (
            extract_boxed_answer(judge_response)
            if len(judge_response) != 1
            else judge_response
        )
        judge_score = 1.0 if judge_result == "A" else 0.0
        self.logger.debug(f"{judge_score=} ({judge_result=})")
        state["judge_result"] = judge_result
        state["judge_score"] = judge_score
        return judge_score

    async def correct_answer(self, state: vf.State, **kwargs) -> float:
        """Whether math verification or judge succeeded."""
        return float(
            state.get("math_verify_score", 0.0) or state.get("judge_score", 0.0)
        )


MATH_VERIFY_SCORER_SCRIPT_TEMPLATE = """\
from pathlib import Path
from math_verify import parse, verify

solution = Path("{solution_path}").read_text()
answer_file = Path("{answer_path}")
answer = answer_file.read_text() if answer_file.exists() else ""

if not answer:
    print(0.0)
else:
    try:
        score = float(
            verify(
                parse("\\\\boxed{{" + solution + "}}", parsing_timeout=5),
                parse("\\\\boxed{{" + answer + "}}", parsing_timeout=5),
                timeout_seconds=5,
            )
        )
        print(score)
    except BaseException:
        print(0.0)
"""


class RemoteHybridMathRubric(SandboxMixin, HybridMathRubric):
    """HybridMathRubric that scores inside the sandbox.

    Expects the environment to keep the sandbox alive for scoring
    (``keep_sandbox_for_scoring=True``) and the agent's answer written to
    ``answer_path``.  Ground-truth is uploaded to ``solution_path`` and a
    scorer script is uploaded and executed.  The sandbox is deleted in the
    ``@vf.cleanup`` handler after scoring.
    """

    DEFAULT_ANSWER_PATH = "/app/answer.txt"
    DEFAULT_SOLUTION_PATH = "/app/solution.txt"
    DEFAULT_SCORER_PATH = "/app/score.py"
    DEFAULT_SCORER_TIMEOUT = 30

    def __init__(
        self,
        answer_path: str = DEFAULT_ANSWER_PATH,
        solution_path: str = DEFAULT_SOLUTION_PATH,
        scorer_path: str = DEFAULT_SCORER_PATH,
        scorer_timeout: int = DEFAULT_SCORER_TIMEOUT,
        sandbox_client_max_workers: int = 50,
        sandbox_client_max_connections: int = 1000,
        sandbox_client_max_keepalive_connections: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.answer_path = answer_path
        self.solution_path = solution_path
        self.scorer_path = scorer_path
        self.scorer_timeout = scorer_timeout
        self.score_script = MATH_VERIFY_SCORER_SCRIPT_TEMPLATE.format(
            answer_path=answer_path,
            solution_path=solution_path,
        )

        self.logger.warning(
            "RemoteHybridMathRubric expects a sandbox kept alive for scoring "
            f"(keep_sandbox_for_scoring=True) and the agent's answer written to {answer_path}"
        )
        self.init_sandbox_client(
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_client_max_connections=sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=sandbox_client_max_keepalive_connections,
        )

    async def math_verify_score(
        self, completion: vf.Messages, answer: str, state: vf.State, **kwargs
    ) -> float:
        """Run math_verify inside the sandbox."""
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            state["math_verify_score"] = 0.0
            return 0.0
        # Track the sandbox so it is torn down on crash
        self.register_sandbox(sandbox_id)
        if state.get("error") or state.get("sandbox_error"):
            state["math_verify_score"] = 0.0
            return 0.0

        try:
            await asyncio.gather(
                self.upload_content(sandbox_id, answer, self.solution_path),
                self.upload_content(sandbox_id, self.score_script, self.scorer_path),
            )
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                f"python3 {self.scorer_path}",
                timeout=self.scorer_timeout,
            )
            if result.exit_code == 0 and result.stdout.strip():
                score = float(result.stdout.strip().splitlines()[-1])
                self.logger.debug(f"Remote math_verify scored {score=}")
            else:
                stderr = (result.stderr or "")[:200]
                self.logger.warning(
                    f"Remote math_verify failed (exit={result.exit_code}): {stderr}"
                )
                score = 0.0
        except Exception as e:
            self.logger.warning(f"Remote math_verify error: {type(e).__name__}: {e}")
            score = 0.0

        state["math_verify_score"] = score
        return score

    async def judge_score(
        self,
        prompt: vf.Messages,
        completion: vf.Messages,
        answer: str,
        state: vf.State,
        **kwargs,
    ) -> float:
        """Judge with response read from the sandbox file."""
        if state.get("math_verify_score", 0) == 1 or self.judge_model is None:
            return state.get("math_verify_score", 0)

        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return 0.0
        response = await self.read_file(sandbox_id, self.answer_path)
        if not response:
            return 0.0
        completion = [vf.AssistantMessage(content=response)]

        judge_response = await self.judge(prompt, completion, answer, state)
        judge_result = (
            extract_boxed_answer(judge_response)
            if len(judge_response) != 1
            else judge_response
        )
        judge_score = 1.0 if judge_result == "A" else 0.0
        self.logger.debug(f"{judge_score=} ({judge_result=})")
        state["judge_result"] = judge_result
        state["judge_score"] = judge_score
        return judge_score

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        """Delete the sandbox after scoring is complete."""
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            await self.delete_sandbox(sandbox_id)
