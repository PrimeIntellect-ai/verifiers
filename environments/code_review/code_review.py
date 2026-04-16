import verifiers as vf
from datasets import Dataset


def load_environment(
    num_train_examples: int = 500,
    num_eval_examples: int = 100,
    **kwargs,
):
    """
    Code Review Environment
    
    Evaluates LLM's ability to review code for bugs, style issues, and improvements.
    """
    
    # Sample code review dataset
    train_data = [
        {
            "prompt": "Review this Python code for bugs and improvements:\n\ndef calculate_average(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total / len(numbers)",
            "answer": "Bug: Division by zero if numbers is empty. Should check len(numbers) > 0 first.",
            "language": "python"
        },
        {
            "prompt": "Review this JavaScript code:\n\nfunction fetchData(url) {\n  fetch(url).then(response => response.json());\n}",
            "answer": "Missing error handling. Should add .catch() for error handling. Also missing return statement.",
            "language": "javascript"
        },
        {
            "prompt": "Review this Python function:\n\ndef process_data(data):\n    result = []\n    for item in data:\n        if item > 0:\n            result.append(item * 2)\n    return result",
            "answer": "Could use list comprehension: return [item * 2 for item in data if item > 0]. More Pythonic.",
            "language": "python"
        },
    ] * (num_train_examples // 3 + 1)
    
    eval_data = [
        {
            "prompt": "Review this code for security issues:\n\nimport os\ndef read_file(filename):\n    return open(filename).read()",
            "answer": "Security issue: No path validation. Could allow directory traversal attacks. Should validate filename.",
            "language": "python"
        },
    ] * (num_eval_examples + 1)
    
    dataset = Dataset.from_list(train_data[:num_train_examples])
    eval_dataset = Dataset.from_list(eval_data[:num_eval_examples])
    
    def score_review(completion, answer):
        """Score code review quality."""
        completion_text = completion[-1]["content"] if completion else ""
        # Simple keyword matching - can be improved
        answer_keywords = set(answer.lower().split())
        completion_keywords = set(completion_text.lower().split())
        overlap = len(answer_keywords & completion_keywords) / len(answer_keywords) if answer_keywords else 0
        return min(1.0, overlap * 2)  # Scale up
    
    rubric = vf.Rubric(funcs=[score_review])
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt="You are an expert code reviewer. Analyze the code and identify bugs, security issues, and potential improvements.",
        rubric=rubric,
        message_type="chat",
    )
