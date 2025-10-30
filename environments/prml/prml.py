import os
import numpy as np

import verifiers as vf
from datasets import load_dataset
from openai import OpenAI
from sympy import simplify, latex
from sympy.parsing.latex import parse_latex


def sympy_simplify_expression(expr_latex: str) -> str:
    try:
        expr = parse_latex(expr_latex)
        simplified = simplify(expr)
        return f"Simplified expression: {latex(simplified)}"
    except Exception as e:
        return f"Error: Could not simplify expression. {str(e)}"


SYSTEM_PROMPT = """You are an expert in Pattern Recognition and Machine Learning. Answer questions from Christopher Bishop's PRML textbook with clear, rigorous mathematical explanations.

**Tools Available:**
You have access to SymPy mathematical verification tools:
1. `sympy_simplify_expression(expr_latex)` - Simplify a mathematical expression

You can use these tools to simplify your expressions during problem-solving.

**Instructions:**
- Provide step-by-step derivations showing all intermediate steps
- Use LaTeX formatting for all mathematical expressions (use $ for inline math, $$ for display math)
- You may use the SymPy tools to simplify your expressions when appropriate
- Clearly state assumptions and definitions when needed
- Work through calculations systematically from the given problem to the final result
- Be precise with mathematical notation, especially indices and exponents
- All the proof and equations and steps should be written in <think> ... </think> tags.
- Finally, output the final answer in the format <answer> ... </answer> tags.

Your answers should be thorough, mathematically sound, and easy to follow."""

judge_prompt = """You are an expert evaluator of mathematical derivations and proofs from Christopher Bishop's "Pattern Recognition and Machine Learning" (PRML). Your task is to assess a student's (base model) response to a PRML exercise against the ground truth solution. These exercises typically involve proving equations, deriving results, or finding expressions .

**Inputs:**
- **Question**: {question}
- **Ground Truth Answer**: {answer} (This is the ideal solution, including key steps, equations, and references.)
- **Model Response**: {response} (The student's attempt, which may include reasoning, derivations, and LaTeX math.)

Evaluate the model response holistically but score each criterion independently. For each, first think step-by-step (write 2-4 sentences of reasoning aloud, referencing specific parts of the response and ground truth). Then, output only the numerical score in the specified XML-like tags. Do not add extra text outside the tags or reasoning.

**Evaluation Criteria:**

1. **Correctness** (Binary: 0 or 1): Does the response logically arrive at the exact final result expected by the question (e.g., the target equation or value)? Award 1 only if the conclusion matches the ground truth's final output precisely (ignoring minor notation differences like vector bolding). Award 0 if the final result is incorrect, absent, or mismatched.
   - Think aloud: Compare the response's endpoint to the question's goal and ground truth.
   - Output: <correctness>1</correctness> or <correctness>0</correctness>

2. **Stepwise Validity** (0.0 to 1.0): Are all steps in the derivation logically sound, with valid substitutions, no mathematical errors, and no unjustified leaps? Score 1.0 for flawless reasoning throughout; deduct proportionally for errors (e.g., 0.5 for one major invalid step, 0.0 for pervasive flaws).
   - Think aloud: Identify 1-2 key steps in the response and verify against ground truth equations/logic.
   - Output: <stepwisevalid>0.8</stepwisevalid> (use one decimal place; e.g., 0.0, 0.5, 1.0)

3. **Readability** (0.0 to 1.0): Is the response clear, well-structured, and concise? Consider logical flow, use of LaTeX for equations, step labeling, and absence of redundancy or jargon. Score 1.0 for textbook-like polish; 0.0 for confusing or rambling text.
   - Think aloud: Note strengths (e.g., "Clear equation chaining") or issues (e.g., "Unlabeled steps obscure flow").
   - Output: <readability>0.9</readability> (use one decimal place)

4. **Completeness** (0.0 to 1.0): Does the response cover all essential elements from the ground truth (e.g., all constraints, substitutions, and intermediate results)? Score 1.0 if nothing critical is omitted; deduct for gaps (e.g., 0.6 if major steps like dual form are skipped).
   - Think aloud: List 1-2 required parts from ground truth and check presence in response.
   - Output: <completeness>0.7</completeness> (use one decimal place)

**Output Format**: After your per-criterion reasoning, provide only the four tagged scores in sequence. No summaries, explanations, or final overall score. Example:
<correctness>1</correctness>
<stepwisevalid>0.9</stepwisevalid>
<readability>0.8</readability>
<completeness>0.9</completeness>
"""

grade_parser_correctness = vf.XMLParser(fields=["correctness"], answer_field="correctness")
grade_parser_clarity_structure = vf.XMLParser(fields=["stepwisevalid"], answer_field="stepwisevalid")
grade_parser_readability = vf.XMLParser(fields=["readability"], answer_field="readability")
grade_parser_completeness = vf.XMLParser(fields=["completeness"], answer_field="completeness")

DIFFICULTY_MULTIPLIERS = {
    "hard": 1.5,
    "medium": 1.0,
    "easy": 0.7,
    "unknown": 1.0
}

def get_difficulty_multiplier(state: dict) -> float:
    difficulty = state.get("difficulty", "unknown")
    if isinstance(difficulty, str):
        difficulty = difficulty.lower()
    return DIFFICULTY_MULTIPLIERS.get(difficulty, 1.0)

async def grade_reward_correctness(judge, prompt, completion, answer, state, **kwargs) -> float:
    judge_response = await judge(prompt, completion, answer, state, **kwargs)
    judge_grade_correctness = grade_parser_correctness.parse_answer(judge_response)
    base_score = int(judge_grade_correctness) if judge_grade_correctness and (int(judge_grade_correctness) == 0 or int(judge_grade_correctness) == 1) else 0
    
    difficulty_mult = get_difficulty_multiplier(state)
    return base_score * difficulty_mult

async def grade_reward_clarity_structure(judge, prompt, completion, answer, state, **kwargs) -> float:
    judge_response = await judge(prompt, completion, answer, state, **kwargs)
    judge_grade_clarity_structure = grade_parser_clarity_structure.parse_answer(judge_response)
    base_score = float(judge_grade_clarity_structure) if judge_grade_clarity_structure and float(judge_grade_clarity_structure) >= 0 and float(judge_grade_clarity_structure) <= 1 else 0
    
    difficulty_mult = get_difficulty_multiplier(state)
    return base_score * difficulty_mult

async def grade_reward_readability(judge, prompt, completion, answer, state, **kwargs) -> float:
    judge_response = await judge(prompt, completion, answer, state, **kwargs)
    judge_grade_readability = grade_parser_readability.parse_answer(judge_response)
    base_score = float(judge_grade_readability) if judge_grade_readability and float(judge_grade_readability) >= 0 and float(judge_grade_readability) <= 1 else 0
    
    difficulty_mult = get_difficulty_multiplier(state)
    return base_score * difficulty_mult

async def grade_reward_completeness(judge, prompt, completion, answer, state, **kwargs) -> float:
    judge_response = await judge(prompt, completion, answer, state, **kwargs)
    judge_grade_completeness = grade_parser_completeness.parse_answer(judge_response)
    base_score = float(judge_grade_completeness) if judge_grade_completeness and float(judge_grade_completeness) >= 0 and float(judge_grade_completeness) <= 1 else 0
    
    difficulty_mult = get_difficulty_multiplier(state)
    return base_score * difficulty_mult

async def grade_reward_similarity(judge, prompt, completion, answer, state, **kwargs) -> float:
    try:
        
        embed_client = kwargs.get('embed_client')
        if embed_client is None:
            
            return 0.5
        
        completion_text = completion if isinstance(completion, str) else (
            completion[0].get('content', '') if isinstance(completion, list) else str(completion)
        )
        answer_text = answer if isinstance(answer, str) else str(answer)
        
        response = embed_client.embeddings.create(
            input=[completion_text, answer_text],
            model=kwargs.get('embed_model', 'text-embedding-3-small')
        )
        
        completion_embedding = np.array(response.data[0].embedding)
        answer_embedding = np.array(response.data[1].embedding)
        
        dot_product = np.dot(completion_embedding, answer_embedding)
        norm_completion = np.linalg.norm(completion_embedding)
        norm_answer = np.linalg.norm(answer_embedding)
        
        cosine_similarity = dot_product / (norm_completion * norm_answer)
        
        similarity_score = max(0.0, min(1.0, cosine_similarity))
        
        difficulty_mult = get_difficulty_multiplier(state)
        return similarity_score * difficulty_mult
        
    except Exception as e:
        print(f"Warning: Similarity calculation failed: {e}")
        return 0.5


def load_environment(
    dataset_name: str = "Vivek/prml-exercises",
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    use_similarity_reward: bool = True,
    use_tools: bool = True,
) -> vf.Environment:
    dataset = load_dataset(dataset_name)
    dataset = dataset['train']
    
    dataset = dataset.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_text']}
        ],
        'answer': x['answer'],
        'difficulty': x['difficulty']
    })
    dataset = dataset.remove_columns(['question_text', 'chapter', 'question_number', 'answer_length'])

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    judge_client = OpenAI(api_key=os.getenv(judge_api_key_var), base_url=judge_base_url)
    
    embed_client = None
    if use_similarity_reward:
        embed_client = OpenAI(api_key=os.getenv(embed_api_key_var), base_url=embed_base_url)
    
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
    )

    rubric.add_reward_func(grade_reward_correctness, weight=2.0)
    rubric.add_reward_func(grade_reward_clarity_structure, weight=2.0)
    rubric.add_reward_func(grade_reward_readability, weight=1.0)
    rubric.add_reward_func(grade_reward_completeness, weight=1.0)
    
    if use_similarity_reward and embed_client:
        async def similarity_reward_wrapper(judge, prompt, completion, answer, state, **kwargs):
            kwargs['embed_client'] = embed_client
            kwargs['embed_model'] = embed_model
            return await grade_reward_similarity(judge, prompt, completion, answer, state, **kwargs)
        
        rubric.add_reward_func(similarity_reward_wrapper, weight=3.0)
    
    if use_tools:
        environment = vf.ToolEnv(
            dataset=train_dataset,
            rubric=rubric,
            tools=[sympy_simplify_expression],
            max_turns=10,
            eval_dataset=eval_dataset,
        )
    else:
        environment = vf.SingleTurnEnv(
            dataset=train_dataset,
            rubric=rubric,
            eval_dataset=eval_dataset,
        )

    return environment