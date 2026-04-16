import verifiers as vf
from datasets import Dataset


def load_environment(
    num_train_examples: int = 300,
    num_eval_examples: int = 50,
    **kwargs,
):
    """
    API Design Environment
    
    Evaluates LLM's ability to design RESTful APIs.
    """
    
    train_data = [
        {
            "prompt": "Design a REST API for a todo list application with CRUD operations.",
            "answer": "Endpoints: GET /todos, POST /todos, GET /todos/:id, PUT /todos/:id, DELETE /todos/:id. Use JSON for request/response bodies.",
            "domain": "productivity"
        },
        {
            "prompt": "Design a REST API for user authentication with JWT tokens.",
            "answer": "Endpoints: POST /auth/register, POST /auth/login, POST /auth/refresh, GET /auth/profile. Use JWT for tokens, bcrypt for passwords.",
            "domain": "security"
        },
    ] * (num_train_examples // 2 + 1)
    
    eval_data = [
        {
            "prompt": "Design a REST API for an e-commerce product catalog.",
            "answer": "Endpoints: GET /products, POST /products, GET /products/:id, PUT /products/:id, DELETE /products/:id, GET /products/search, GET /categories.",
            "domain": "e-commerce"
        },
    ] * (num_eval_examples + 1)
    
    dataset = Dataset.from_list(train_data[:num_train_examples])
    eval_dataset = Dataset.from_list(eval_data[:num_eval_examples])
    
    def score_api_design(completion, answer):
        """Score API design quality."""
        completion_text = completion[-1]["content"] if completion else ""
        # Check for key API elements
        score = 0.0
        if "endpoint" in completion_text.lower() or "route" in completion_text.lower():
            score += 0.3
        if "get" in completion_text.lower() and "post" in completion_text.lower():
            score += 0.3
        if "json" in completion_text.lower():
            score += 0.2
        if ":" in completion_text:  # Path parameters
            score += 0.2
        return min(1.0, score)
    
    rubric = vf.Rubric(funcs=[score_api_design])
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt="You are an API design expert. Design clean, RESTful APIs following best practices.",
        rubric=rubric,
        message_type="chat",
    )
