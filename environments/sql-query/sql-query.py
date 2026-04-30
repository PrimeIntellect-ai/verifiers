import verifiers as vf
from datasets import Dataset


def load_environment(
    num_train_examples: int = 400,
    num_eval_examples: int = 80,
    **kwargs,
):
    """
    SQL Query Environment
    
    Evaluates LLM's ability to write SQL queries.
    """
    
    train_data = [
        {
            "prompt": "Write a SQL query to find all users who registered in the last 30 days.",
            "answer": "SELECT * FROM users WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY);",
            "difficulty": "easy"
        },
        {
            "prompt": "Write a SQL query to find the top 5 customers by total order amount.",
            "answer": "SELECT c.name, SUM(o.amount) as total FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id ORDER BY total DESC LIMIT 5;",
            "difficulty": "medium"
        },
    ] * (num_train_examples // 2 + 1)
    
    eval_data = [
        {
            "prompt": "Write a SQL query to find products that have never been ordered.",
            "answer": "SELECT p.* FROM products p LEFT JOIN order_items oi ON p.id = oi.product_id WHERE oi.product_id IS NULL;",
            "difficulty": "medium"
        },
    ] * (num_eval_examples + 1)
    
    dataset = Dataset.from_list(train_data[:num_train_examples])
    eval_dataset = Dataset.from_list(eval_data[:num_eval_examples])
    
    def score_sql(completion, answer):
        """Score SQL query quality."""
        completion_text = completion[-1]["content"] if completion else ""
        score = 0.0
        if "select" in completion_text.lower():
            score += 0.3
        if "from" in completion_text.lower():
            score += 0.2
        if "where" in completion_text.lower() or "join" in completion_text.lower():
            score += 0.3
        if "group by" in completion_text.lower() or "order by" in completion_text.lower():
            score += 0.2
        return min(1.0, score)
    
    rubric = vf.Rubric(funcs=[score_sql])
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt="You are a SQL expert. Write efficient, correct SQL queries.",
        rubric=rubric,
        message_type="chat",
    )
