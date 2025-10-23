"""
Example demonstrating multi-turn training with VerifiersGRPOTrainer.

This example shows how to train a model on multi-turn conversations with tool usage,
using the new backwards-compatible multi-turn extension.
"""

from datasets import Dataset
from transformers import AutoTokenizer

from verifiers.envs import ToolEnv
from verifiers.trainers import VerifiersGRPOTrainer, VerifiersGRPOConfig


def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform basic arithmetic operations.
    
    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        The result of the operation
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def get_weather(location: str) -> str:
    """
    Get weather information for a location.
    
    Args:
        location: The location to get weather for
        
    Returns:
        Weather description
    """
    # Mock weather function
    return f"The weather in {location} is sunny and 72°F"


def main():
    # Create sample dataset for multi-turn training
    dataset = Dataset.from_dict({
        "question": [
            "What's 15 + 27? Then tell me the weather in Paris.",
            "Calculate 100 / 4 and then get weather for Tokyo.",
            "Multiply 8 by 7, then check weather in London.",
        ],
        "answer": [
            "42, sunny and 72°F",
            "25.0, sunny and 72°F", 
            "56, sunny and 72°F",
        ]
    })
    
    # System prompt that explains tool usage
    system_prompt = """You are a helpful assistant that can perform calculations and check weather.

You have access to the following tools:
{tool_descriptions}

When you need to perform calculations or check weather, use the appropriate tools.
Format your tool calls as XML: <tool>{{"name": "tool_name", "args": {{"arg1": "value1"}}}}</tool>

Always provide your final answer after using tools."""

    # Create multi-turn environment with tools
    env = ToolEnv(
        tools=[calculator, get_weather],
        dataset=dataset,
        system_prompt=system_prompt,
        format_prompt=True,  # Automatically format tool descriptions
        max_turns=10,
    )
    
    # Print detected environment type
    print(f"Environment type: {type(env).__name__}")
    
    # Create trainer configuration
    config = VerifiersGRPOConfig(
        output_dir="./multi_turn_output",
        run_name="multi-turn-tool-example",
        learning_rate=1e-6,
        per_device_train_batch_size=2,  # Small batch for example
        num_generations=2,
        max_steps=10,  # Few steps for example
        bf16=False,  # Disable for compatibility
        fp16=False,
        save_strategy="steps",
        save_steps=5,
        logging_steps=1,
        # Multi-turn specific settings
        enable_async_generation=False,  # Disable for simplicity
        num_batches_ahead=0,
    )
    
    # Create tokenizer (replace with your model's tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained("your-model-name")
    # For this example, we'll skip actual training
    
    print("Creating VerifiersGRPOTrainer...")
    
    # Create trainer - will automatically detect multi-turn environment
    trainer = VerifiersGRPOTrainer(
        model="microsoft/DialoGPT-small",  # Example model
        env=env,
        args=config,
        # processing_class=tokenizer,
    )
    
    print(f"Multi-turn detection: {trainer.is_multi_turn}")
    print("Multi-turn environment detected!" if trainer.is_multi_turn else "Single-turn environment detected!")
    
    # Show what a sample rollout would look like
    print("\n=== Sample Multi-Turn Conversation ===")
    
    # This is what the environment would generate during training:
    # Get the formatted system prompt (this happens automatically in ToolEnv)
    tool_descriptions = "calculator: Perform basic arithmetic operations\nget_weather: Get weather information for a location"
    formatted_system_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
    
    sample_prompt = [
        {"role": "system", "content": formatted_system_prompt},
        {"role": "user", "content": "What's 15 + 27? Then tell me the weather in Paris."}
    ]
    
    print("Initial prompt:")
    for msg in sample_prompt:
        print(f"  {msg['role']}: {msg['content'][:100]}...")
    
    print("\nMulti-turn conversation would include:")
    print("  1. Assistant: <tool>{\"name\": \"calculator\", \"args\": {...}}</tool>")
    print("  2. User: <result>42</result>")
    print("  3. Assistant: <tool>{\"name\": \"get_weather\", \"args\": {...}}</tool>") 
    print("  4. User: <result>The weather in Paris is sunny and 72°F</result>")
    print("  5. Assistant: The answer is 42, and the weather in Paris is sunny and 72°F")
    
    print("\n=== Training Configuration ===")
    print(f"Trainer type: {type(trainer).__name__}")
    print(f"Environment type: {type(trainer.env).__name__}")
    print(f"Multi-turn support: {trainer.is_multi_turn}")
    print(f"Max turns: {env.max_turns}")
    print(f"Available tools: {[tool.__name__ for tool in env.tools]}")
    
    # Uncomment to actually train:
    # print("\nStarting training...")
    # trainer.train()
    
    print("\nExample completed! Uncomment trainer.train() to actually train the model.")


if __name__ == "__main__":
    main()